import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
import pandas as pd
from collections import defaultdict
from itertools import combinations
import socket
import struct

from psignal.lofar import norm, spec_single_stft_log10
from testfile.extract_multi_lines import extract_multi_lines
from yolo_denoiser import YoloDenoiser


# ============================================================
# 1. TCP接收函数
# ============================================================

def receive_tcp_data(sock):
    """接收1秒TCP数据并解析

    返回: (fs, pt_data, vx_data, vy_data)
    """
    # 先接收4字节采样率
    fs_bytes = b''
    while len(fs_bytes) < 4:
        chunk = sock.recv(4 - len(fs_bytes))
        if not chunk:
            raise ConnectionError("TCP connection closed")
        fs_bytes += chunk

    fs = struct.unpack('>i', fs_bytes)[0]

    # 计算剩余字节数: 3*Fs*4(三通道数据)
    remaining_bytes = 3 * fs * 4

    # 接收三通道数据
    data = b''
    while len(data) < remaining_bytes:
        chunk = sock.recv(remaining_bytes - len(data))
        if not chunk:
            raise ConnectionError("TCP connection closed")
        data += chunk

    # 解析三通道数据 (大端序单精度浮点数)
    offset = 0
    pt_data = np.array(struct.unpack(f'>{fs}f', data[offset:offset + fs*4]), dtype=np.float32)
    offset += fs * 4
    vx_data = np.array(struct.unpack(f'>{fs}f', data[offset:offset + fs*4]), dtype=np.float32)
    offset += fs * 4
    vy_data = np.array(struct.unpack(f'>{fs}f', data[offset:offset + fs*4]), dtype=np.float32)

    return fs, pt_data, vx_data, vy_data


# ============================================================
# 2. 圆周角度工具函数
# ============================================================

def angular_diff_deg(a, b):
    """最小圆周角差，范围 [0, 180]"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.abs(a - b) % 360.0
    return np.minimum(diff, 360.0 - diff)


def circular_signed_diff_deg(a, b):
    """返回 a-b 的有符号最小角差，范围 [-180, 180)"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return ((a - b + 180.0) % 360.0) - 180.0


def circular_mean_deg(angles_deg):
    """圆周均值（单位：度）"""
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if len(angles_deg) == 0:
        return np.nan
    ang = np.deg2rad(angles_deg)
    s = np.mean(np.sin(ang))
    c = np.mean(np.cos(ang))
    mean_ang = np.rad2deg(np.arctan2(s, c)) % 360.0
    return float(mean_ang)


def circular_smooth_deg(angles_deg, win=5):
    """对DOA做滑动圆周平滑（向量化版本，与逐点版本行为一致）"""
    x = np.asarray(angles_deg, dtype=float)
    n = len(x)
    if n == 0 or win <= 1:
        return x.copy()
    half = win // 2
    ang = np.deg2rad(x)
    s = np.sin(ang)
    c = np.cos(ang)

    # 累加和实现滑动均值，边界处自动缩小窗口
    cs = np.concatenate([[0], np.cumsum(s)])
    cc = np.concatenate([[0], np.cumsum(c)])

    indices = np.arange(n)
    lefts = np.maximum(0, indices - half)
    rights = np.minimum(n, indices + half + 1)
    counts = (rights - lefts).astype(float)

    s_mean = (cs[rights] - cs[lefts]) / counts
    c_mean = (cc[rights] - cc[lefts]) / counts

    return np.rad2deg(np.arctan2(s_mean, c_mean)) % 360.0


def remove_doa_outliers(angles_deg, win=5, thr=20.0, max_iter=2):
    """迭代式局部异常点检测（向量化版本）"""
    x = np.asarray(angles_deg, dtype=float).copy()
    n = len(x)
    if n == 0:
        return x
    for _ in range(max_iter):
        smoothed = circular_smooth_deg(x, win=win)
        diffs = angular_diff_deg(x, smoothed)
        outliers = diffs > thr
        if not np.any(outliers):
            break
        x[outliers] = smoothed[outliers]
    return x


# ============================================================
# 2. 基础函数（从原文件复制）
# ============================================================

def wave_to_spec(wave, frequency_resolution, fs, f_lower_bound, f_higher_bound):
    """将波形转换为时频图"""
    f_lower_bound = int(f_lower_bound)
    f_higher_bound = int(f_higher_bound)
    assert len(wave.shape) == 2
    wave = norm(wave)
    spec = spec_single_stft_log10(wave, f_lower_bound, f_higher_bound, frequency_resolution, fs)
    spec = norm(spec)
    return spec


def to_image(data):
    """谱图转图像"""
    data = np.asarray(data)
    if data.ndim == 3:
        img = data.copy()
        if img.dtype != np.uint8:
            dmin = np.nanmin(img)
            dmax = np.nanmax(img)
            if not np.isfinite(dmin) or not np.isfinite(dmax) or abs(dmax - dmin) < 1e-12:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = ((img - dmin) * 255.0 / (dmax - dmin)).astype(np.uint8)
        return img

    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    if not np.isfinite(dmin) or not np.isfinite(dmax) or abs(dmax - dmin) < 1e-12:
        img = np.zeros_like(data, dtype=np.uint8)
    else:
        img = ((data - dmin) * 255.0 / (dmax - dmin)).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


def make_frequency_ruler(width, f_lower_bound, f_higher_bound, height=40):
    """生成频率刻度尺"""
    ruler = np.zeros((height, width, 3), dtype=np.uint8)
    if width <= 0:
        return ruler
    f0 = float(f_lower_bound)
    f1 = float(f_higher_bound)
    if f1 <= f0:
        return ruler
    freq_range = f1 - f0
    major_tick = 10
    minor_tick = major_tick / 2.0

    # 小刻度
    f = np.ceil(f0 / minor_tick) * minor_tick
    while f <= f1:
        x = int(round((f - f0) / freq_range * (width - 1)))
        cv2.line(ruler, (x, height - 12), (x, height - 1), (120, 120, 120), 1)
        f += minor_tick

    # 大刻度
    f = np.ceil(f0 / major_tick) * major_tick
    while f <= f1:
        x = int(round((f - f0) / freq_range * (width - 1)))
        cv2.line(ruler, (x, height - 26), (x, height - 1), (255, 255, 255), 1)
        text = f"{int(f)}Hz"
        cv2.putText(ruler, text, (min(x + 2, width - 60), 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        f += major_tick
    return ruler


def test_deepdenoiser_stft_fragment(model, noise_spec, device='cpu', thresh=0.1, line_channel=0):
    """对单张时频图做深度模型去噪"""
    ori_h, ori_w = noise_spec.shape

    # 计算pad后的尺寸：高度多pad 32行
    target_h = (ori_h - 1) // 32 + 1
    target_w = (ori_w - 1) // 32 + 1
    pad_h = target_h * 32 + 32
    pad_w = target_w * 32

    # 创建pad后的数组
    padded = np.zeros((pad_h, pad_w), dtype=np.float32)

    # 先复制原始数据
    padded[:ori_h, :ori_w] = noise_spec.astype(np.float32)

    # 高度方向：用最后rows_to_pad行填充
    if pad_h > ori_h:
        rows_to_pad = pad_h - ori_h
        padded[ori_h:pad_h, :ori_w] = noise_spec[-rows_to_pad:][::-1]

    # 宽度方向：用最后几列填充
    if pad_w > ori_w:
        cols_to_pad = pad_w - ori_w
        padded[:ori_h, ori_w:pad_w] = noise_spec[:, -cols_to_pad:]
        # 右下角也填充
        if pad_h > ori_h:
            rows_to_pad = pad_h - ori_h
            padded[ori_h:pad_h, ori_w:pad_w] = noise_spec[-rows_to_pad:, -cols_to_pad:]

    x = torch.tensor(padded, dtype=torch.float32)
    x = x.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        res = model(x, flag='test')
        sf = nn.Softmax(dim=1).to(device)
        res = sf(res)
        res = res.detach().cpu().numpy()[0]
        conf = res[line_channel]

        # 只取原始尺寸的部分，去掉所有pad
        conf = conf[:ori_h, :ori_w]

        conf_soft = conf.copy()
        conf_soft[:, :2000] = np.where(conf_soft[:, :2000] < thresh, 0, conf_soft[:, :2000])
        conf_soft[:, 2000:] = np.where(conf_soft[:, 2000:] < thresh, 0, conf_soft[:, 2000:])
        foreground = conf_soft * noise_spec

        conf_bin = conf_soft.copy()
        conf_bin[conf_bin < thresh] = 0
        conf_bin[conf_bin >= thresh] = 1
        conf_bin = conf_bin.astype(np.int32)

    return foreground, conf_bin


def _denoise_one_tile(args):
    """单块去噪（供 ThreadPoolExecutor 调用）"""
    model, tile, device, thresh, line_channel = args
    return test_deepdenoiser_stft_fragment(model, tile, device=device,
                                           thresh=thresh, line_channel=line_channel)


def test_deepdenoiser_stft_fragment_tiled(model, noise_spec, device='cpu',
                                           thresh=0.1, line_channel=0,
                                           tile_width=5000, tile_overlap=256,
                                           max_workers=4):
    """分块并行去噪 — 沿频率轴切分，ThreadPoolExecutor 并行"""
    from concurrent.futures import ThreadPoolExecutor

    ori_h, ori_w = noise_spec.shape

    if ori_w <= tile_width + tile_overlap:
        return test_deepdenoiser_stft_fragment(model, noise_spec, device=device,
                                               thresh=thresh, line_channel=line_channel)

    tile_starts = list(range(0, ori_w, tile_width - tile_overlap))
    if tile_starts[-1] + tile_width < ori_w:
        tile_starts.append(ori_w - tile_width)
    tile_starts = sorted(set(s for s in tile_starts if s >= 0 and s + tile_width <= ori_w))

    tiles = [noise_spec[:, s:s + tile_width] for s in tile_starts]

    args_list = [(model, tile, device, thresh, line_channel) for tile in tiles]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tiles))) as executor:
        results = list(executor.map(_denoise_one_tile, args_list))

    foreground_full = np.zeros_like(noise_spec, dtype=np.float32)
    conf_full = np.zeros_like(noise_spec, dtype=np.int32)

    for s, (fg, cf) in zip(tile_starts, results):
        e = s + tile_width
        if s > 0:
            overlap_start = s
            overlap_end = s + tile_overlap
            w = np.linspace(0, 1, overlap_end - overlap_start).astype(np.float32)
            foreground_full[:, overlap_start:overlap_end] = (
                foreground_full[:, overlap_start:overlap_end] * (1 - w) +
                fg[:, overlap_start - s:overlap_start - s + tile_overlap] * w
            )
            conf_full[:, overlap_start:overlap_end] = cf[:, overlap_start - s:overlap_start - s + tile_overlap]
        else:
            foreground_full[:, :tile_width] = fg
            conf_full[:, :tile_width] = cf

        if s + tile_overlap < e:
            non_overlap_start = s + tile_overlap if s > 0 else tile_width
            non_overlap_end = e
            fg_start = non_overlap_start - s
            fg_end = non_overlap_end - s
            if fg_end > fg_start:
                foreground_full[:, non_overlap_start:non_overlap_end] = fg[:, fg_start:fg_end]
                conf_full[:, non_overlap_start:non_overlap_end] = cf[:, fg_start:fg_end]

    return foreground_full, conf_full


def vector_doa_batch(pt_buffer, vx_buffer, vy_buffer, fs, buffer_start_time,
                     points_list, win_len=10, mode='center', device='cpu'):
    """批量DOA估计 - 优化版本"""
    import time
    t_total = time.time()

    if len(points_list) == 0:
        return []

    pt_buffer = np.asarray(pt_buffer, dtype=float).flatten()
    vx_buffer = np.asarray(vx_buffer, dtype=float).flatten()
    vy_buffer = np.asarray(vy_buffer, dtype=float).flatten()

    # 缓冲区级别归一化（逼近全局归一化效果）
    pt_buffer = pt_buffer / (np.abs(pt_buffer).max() + 1e-10)
    vx_buffer = vx_buffer / (np.abs(vx_buffer).max() + 1e-10)
    vy_buffer = vy_buffer / (np.abs(vy_buffer).max() + 1e-10)

    current_time = buffer_start_time + len(pt_buffer) / fs

    nfft = int(win_len * fs)
    half_nfft = nfft // 2

    # 按时间窗口分组
    time_groups = {}
    for idx, (target_time, target_freq) in enumerate(points_list):
        if mode == 'center':
            t1 = target_time - win_len / 2.0
            t2 = target_time + win_len / 2.0
        else:  # causal
            t1 = target_time - win_len
            t2 = target_time

        if t1 < buffer_start_time or t2 > current_time:
            continue

        start_idx = int(round((t1 - buffer_start_time) * fs))
        end_idx = int(round((t2 - buffer_start_time) * fs))
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(pt_buffer))

        if end_idx <= start_idx:
            continue

        sig_len = end_idx - start_idx
        if sig_len < int(win_len * fs * 0.9):
            continue

        key = (start_idx, end_idx)
        if key not in time_groups:
            time_groups[key] = []
        time_groups[key].append((idx, target_freq))

    results = [None] * len(points_list)

    # 对每个时间窗口批量处理
    for (start_idx, end_idx), freq_list in time_groups.items():
        p = torch.from_numpy(pt_buffer[start_idx:end_idx]).float().to(device)
        x = torch.from_numpy(vx_buffer[start_idx:end_idx]).float().to(device)
        y = torch.from_numpy(vy_buffer[start_idx:end_idx]).float().to(device)

        sig_len = len(p)
        win = torch.from_numpy(np.hanning(sig_len)).float().to(device)

        # 批量FFT（rfft 利用实数对称性，约快2倍）
        signals = torch.stack([p * win, x * win, y * win])
        ffts = torch.fft.rfft(signals, n=nfft, dim=1) / sig_len

        # 对每个频率点计算DOA
        for idx, target_freq in freq_list:
            f_low = max(target_freq - 0.5, 10)
            f_high = target_freq + 0.5
            f_ln = max(1, int(np.floor(f_low * nfft / fs + 0.5)))
            f_hn = min(half_nfft, int(np.floor(f_high * nfft / fs + 0.5)))

            if f_ln > f_hn:
                results[idx] = -5
                continue

            p_seg = ffts[0, f_ln - 1:f_hn]
            x_seg = ffts[1, f_ln - 1:f_hn]
            y_seg = ffts[2, f_ln - 1:f_hn]

            if len(p_seg) == 0:
                results[idx] = -5
                continue

            # 交叉谱和角度估计
            Pvx2 = torch.real(p_seg * torch.conj(x_seg))
            Pvy2 = torch.real(p_seg * torch.conj(y_seg))
            est_angle = torch.atan2(Pvy2, Pvx2) * 180 / np.pi

            # unwrap和mod
            est_angle_np = est_angle.cpu().numpy()
            theta_unwrap_rad = np.unwrap(np.radians(est_angle_np))
            est_angle_np = np.mod(np.degrees(theta_unwrap_rad) + 180, 360)
            est_angle = torch.from_numpy(est_angle_np).float().to(device)

            # 幅度加权
            amp = torch.abs(p_seg)
            if torch.all(amp == 0) or torch.max(amp) <= 1e-10:
                results[idx] = -5
                continue

            weight = amp / torch.max(amp)
            nbins = 720
            theta_edges = torch.linspace(0, 360, nbins + 1, device=device)
            theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0

            bin_idx = torch.searchsorted(theta_edges, est_angle, right=False)
            bin_idx = torch.clamp(bin_idx, 0, nbins - 1)

            valid = torch.isfinite(est_angle) & torch.isfinite(weight) & (bin_idx >= 0) & (bin_idx < nbins)

            if not torch.any(valid):
                results[idx] = -5
                continue

            count_weight = torch.zeros(nbins, dtype=torch.float32, device=device)
            count_weight.scatter_add_(0, bin_idx[valid], weight[valid])

            max_idx = torch.argmax(count_weight)
            doa = theta_centers[max_idx].item()
            results[idx] = doa

    return results


def vector_doa_one_point(pt_buffer, vx_buffer, vy_buffer, fs, buffer_start_time,
                         target_time, target_freq, win_len=10, mode='center', device='cpu', debug=False):
    """单点 DOA 估计 (Torch优化版本)"""
    import time
    t_total = time.time()

    pt_buffer = np.asarray(pt_buffer, dtype=float).flatten()
    vx_buffer = np.asarray(vx_buffer, dtype=float).flatten()
    vy_buffer = np.asarray(vy_buffer, dtype=float).flatten()

    # 缓冲区级别归一化（逼近全局归一化效果）
    pt_buffer = pt_buffer / (np.abs(pt_buffer).max() + 1e-10)
    vx_buffer = vx_buffer / (np.abs(vx_buffer).max() + 1e-10)
    vy_buffer = vy_buffer / (np.abs(vy_buffer).max() + 1e-10)

    current_time = buffer_start_time + len(pt_buffer) / fs

    nfft = int(win_len * fs)
    half_nfft = nfft // 2

    if mode == 'center':
        t1 = target_time - win_len / 2.0
        t2 = target_time + win_len / 2.0
    elif mode == 'causal':
        t1 = target_time - win_len
        t2 = target_time
    else:
        raise ValueError("mode must be 'center' or 'causal'")

    if t1 < buffer_start_time or t2 > current_time:
        return None

    start_idx = int(round((t1 - buffer_start_time) * fs))
    end_idx = int(round((t2 - buffer_start_time) * fs))
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(pt_buffer))

    if end_idx <= start_idx:
        return None

    sig_len = end_idx - start_idx
    if sig_len < int(win_len * fs * 0.9):
        return None

    # 转换为torch张量
    t0 = time.time()
    p = torch.from_numpy(pt_buffer[start_idx:end_idx]).float().to(device)
    x = torch.from_numpy(vx_buffer[start_idx:end_idx]).float().to(device)
    y = torch.from_numpy(vy_buffer[start_idx:end_idx]).float().to(device)
    time_to_tensor = time.time() - t0

    sig_len = len(p)

    t2 = time.time()
    win = torch.from_numpy(np.hanning(sig_len)).float().to(device)
    time_window = time.time() - t2

    # FFT批量处理（rfft）
    t3 = time.time()
    signals = torch.stack([p * win, x * win, y * win])
    ffts = torch.fft.rfft(signals, n=nfft, dim=1) / sig_len
    time_fft = time.time() - t3

    def matlab_round_pos(v):
        return int(np.floor(v + 0.5))

    f_low = max(target_freq - 0.5, 10)
    f_high = target_freq + 0.5
    f_ln = max(1, matlab_round_pos(f_low * nfft / fs))
    f_hn = min(half_nfft, matlab_round_pos(f_high * nfft / fs))

    if f_ln > f_hn:
        return None

    p_seg = ffts[0, f_ln - 1:f_hn]
    x_seg = ffts[1, f_ln - 1:f_hn]
    y_seg = ffts[2, f_ln - 1:f_hn]

    if len(p_seg) == 0:
        return None

    # 交叉谱和角度估计
    t4 = time.time()
    Pvx2 = torch.real(p_seg * torch.conj(x_seg))
    Pvy2 = torch.real(p_seg * torch.conj(y_seg))
    est_angle = torch.atan2(Pvy2, Pvx2) * 180 / np.pi
    time_cross_spectrum = time.time() - t4

    # unwrap和mod
    t5 = time.time()
    est_angle_np = est_angle.cpu().numpy()
    theta_unwrap_rad = np.unwrap(np.radians(est_angle_np))
    est_angle_np = np.mod(np.degrees(theta_unwrap_rad) + 180, 360)
    est_angle = torch.from_numpy(est_angle_np).float().to(device)
    time_unwrap = time.time() - t5

    # 幅度加权
    t6 = time.time()
    amp = torch.abs(p_seg)
    if torch.all(amp == 0) or torch.max(amp) <= 1e-10:
        return -5

    weight = amp / torch.max(amp)
    nbins = 720
    theta_edges = torch.linspace(0, 360, nbins + 1, device=device)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0

    bin_idx = torch.searchsorted(theta_edges, est_angle, right=False)
    bin_idx = torch.clamp(bin_idx, 0, nbins - 1)

    valid = torch.isfinite(est_angle) & torch.isfinite(weight) & (bin_idx >= 0) & (bin_idx < nbins)

    if not torch.any(valid):
        return -5

    count_weight = torch.zeros(nbins, dtype=torch.float32, device=device)
    count_weight.scatter_add_(0, bin_idx[valid], weight[valid])

    max_idx = torch.argmax(count_weight)
    doa = theta_centers[max_idx].item()
    time_histogram = time.time() - t6

    time_total_elapsed = time.time() - t_total

    if debug:
        print(f"  [DOA_DEBUG] total={time_total_elapsed*1000:.2f}ms "
              f"(tensor={time_to_tensor*1000:.2f}ms "
              f"win={time_window*1000:.2f}ms fft={time_fft*1000:.2f}ms "
              f"cross={time_cross_spectrum*1000:.2f}ms unwrap={time_unwrap*1000:.2f}ms "
              f"hist={time_histogram*1000:.2f}ms)")

    return doa


# ============================================================
# 3. 图连通分量
# ============================================================

def connected_components(graph):
    """查找图的连通分量"""
    visited = set()
    comps = []
    for node in graph:
        if node in visited:
            continue
        stack = [node]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nb in graph[cur]:
                if nb not in visited:
                    stack.append(nb)
        comps.append(sorted(comp))
    return comps


# ============================================================
# 4. 轨迹时间窗口过滤
# ============================================================

def filter_tracks_by_time_window(tracks, current_time, window_seconds=1200):
    """只保留最近window_seconds内有数据的轨迹"""
    cutoff_time = current_time - window_seconds
    filtered = {}
    for tid, tr in tracks.items():
        if tr['t_end'] >= cutoff_time:
            filtered[tid] = tr
    return filtered


# ============================================================
# 5. 实时聚类分析器
# ============================================================

class RealtimeClusterAnalyzer:
    """实时轨迹聚类分析器，每20秒更新一次"""

    def __init__(
        self,
        analysis_window=600,  # 分析窗口：10分钟
        doa_outlier_win=5,
        doa_outlier_thr=20.0,
        doa_smooth_win=5,
        min_overlap_points=120,
        doa_mean_thr=30.0,
        doa_median_thr=30.0,
        min_abs_doa_slope=0.01,
        max_doa_disp=25.0,
        min_cluster_size=3,
        k_core=2
    ):
        self.analysis_window = analysis_window
        self.doa_outlier_win = doa_outlier_win
        self.doa_outlier_thr = doa_outlier_thr
        self.doa_smooth_win = doa_smooth_win
        self.min_overlap_points = min_overlap_points
        self.doa_mean_thr = doa_mean_thr
        self.doa_median_thr = doa_median_thr
        self.min_abs_doa_slope = min_abs_doa_slope
        self.max_doa_disp = max_doa_disp
        self.min_cluster_size = min_cluster_size
        self.k_core = k_core

        # 存储所有轨迹数据
        self.all_tracks = {}  # {track_id: {'time': [], 'freq': [], 'doa': [], 't_start': float, 't_end': float}}

    def update_tracks(self, new_points):
        """更新轨迹数据"""
        for p in new_points:
            tid = p['track_id']
            if tid not in self.all_tracks:
                self.all_tracks[tid] = {
                    'time': [],
                    'freq': [],
                    'doa': [],
                    't_start': p['time'],
                    't_end': p['time']
                }
            tr = self.all_tracks[tid]
            tr['time'].append(p['time'])
            tr['freq'].append(p['freq'])
            tr['doa'].append(p['doa'])
            tr['t_end'] = p['time']

    def analyze(self, current_time):
        """执行实时分析"""
        # 1. 过滤最近20分钟的轨迹
        active_tracks = filter_tracks_by_time_window(
            self.all_tracks, current_time, self.analysis_window
        )

        if len(active_tracks) == 0:
            return {
                'track_count': 0,
                'cluster_count': 0,
                'pair_count': 0,
                'consistent_pair_count': 0
            }

        # 2. DOA预处理
        processed_tracks = {}
        for tid, tr in active_tracks.items():
            if len(tr['time']) == 0:
                continue
            t = np.array(tr['time'])
            f = np.array(tr['freq'])
            doa = np.array(tr['doa'])

            # 异常点检测和平滑
            doa_clean = remove_doa_outliers(doa, self.doa_outlier_win, self.doa_outlier_thr)
            doa_smooth = circular_smooth_deg(doa_clean, self.doa_smooth_win)

            processed_tracks[tid] = {
                'time': t,
                'freq': f,
                'doa_smooth': doa_smooth,
                't_start': tr['t_start'],
                't_end': tr['t_end']
            }

        # 3. 特征提取和过滤
        valid_tracks = {}
        for tid, tr in processed_tracks.items():
            # 计算DOA斜率和离散度
            doa_slope = self._fit_doa_slope(tr['time'], tr['doa_smooth'])
            doa_disp = self._circular_dispersion(tr['doa_smooth'])

            # 过滤
            if abs(doa_slope) >= self.min_abs_doa_slope and doa_disp <= self.max_doa_disp:
                valid_tracks[tid] = tr

        if len(valid_tracks) < 2:
            return {
                'track_count': len(active_tracks),
                'valid_track_count': len(valid_tracks),
                'cluster_count': len(valid_tracks),
                'pair_count': 0,
                'consistent_pair_count': 0,
                'clusters': []
            }

        # 4. 两两相似度计算（向量化）
        pair_results, consistent_count = self._compute_pairwise_similarity_vectorized(valid_tracks)

        # 5. 图聚类
        graph = {tid: set() for tid in valid_tracks.keys()}
        for pr in pair_results:
            if pr['consistent']:
                graph[pr['tid1']].add(pr['tid2'])
                graph[pr['tid2']].add(pr['tid1'])

        # k-core过滤
        if self.k_core is not None:
            graph = self._apply_k_core(graph, self.k_core)

        comps = connected_components(graph)

        # 过滤小簇
        comps = [c for c in comps if len(c) >= self.min_cluster_size]

        return {
            'track_count': len(active_tracks),
            'valid_track_count': len(valid_tracks),
            'cluster_count': len(comps),
            'pair_count': len(pair_results),
            'consistent_pair_count': consistent_count,
            'clusters': comps
        }

    def _apply_k_core(self, graph, k):
        """应用k-core过滤"""
        graph = {node: neighbors.copy() for node, neighbors in graph.items()}
        changed = True
        while changed:
            changed = False
            to_remove = []
            for node in graph:
                if len(graph[node]) < k:
                    to_remove.append(node)
            if to_remove:
                changed = True
                for node in to_remove:
                    for neighbor in graph[node]:
                        if neighbor in graph:
                            graph[neighbor].discard(node)
                    del graph[node]
        return graph

    def _fit_doa_slope(self, time, doa_deg):
        """拟合DOA斜率"""
        if len(time) < 2:
            return 0.0
        doa_rad = np.deg2rad(doa_deg)
        doa_unwrap = np.unwrap(doa_rad)
        doa_unwrap_deg = np.rad2deg(doa_unwrap)
        p = np.polyfit(time, doa_unwrap_deg, 1)
        return float(p[0])

    def _circular_dispersion(self, angles_deg):
        """圆周离散度"""
        mean = circular_mean_deg(angles_deg)
        diffs = angular_diff_deg(angles_deg, mean)
        return float(np.mean(diffs))

    def _compute_doa_similarity(self, tr1, tr2):
        """计算两条轨迹的DOA相似度"""
        t1, d1 = tr1['time'], tr1['doa_smooth']
        t2, d2 = tr2['time'], tr2['doa_smooth']

        # 找重叠时间段
        t_start = max(np.min(t1), np.min(t2))
        t_end = min(np.max(t1), np.max(t2))

        if t_end <= t_start:
            return {'overlap_points': 0, 'consistent': False}

        # 重采样
        if len(t1) < 2 or len(t2) < 2:
            return {'overlap_points': 0, 'consistent': False}
        dt1 = np.median(np.diff(t1))
        dt2 = np.median(np.diff(t2))
        if not (np.isfinite(dt1) and np.isfinite(dt2) and dt1 > 0 and dt2 > 0):
            return {'overlap_points': 0, 'consistent': False}
        dt = min(dt1, dt2)
        target_times = np.arange(t_start, t_end, dt)

        d1i = np.interp(target_times, t1, d1)
        d2i = np.interp(target_times, t2, d2)

        if len(target_times) < self.min_overlap_points:
            return {'overlap_points': len(target_times), 'consistent': False}

        # 计算差异
        diffs = np.abs(circular_signed_diff_deg(d1i, d2i))
        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))

        consistent = (mean_diff <= self.doa_mean_thr) and (median_diff <= self.doa_median_thr)

        return {
            'overlap_points': len(target_times),
            'mean_diff': mean_diff,
            'median_diff': median_diff,
            'consistent': consistent
        }

    def _compute_pairwise_similarity_vectorized(self, valid_tracks):
        """向量化两两相似度计算"""
        tid_list = list(valid_tracks.keys())
        n_tracks = len(tid_list)
        if n_tracks < 2:
            return [], 0

        track_data = []
        for tid in tid_list:
            tr = valid_tracks[tid]
            track_data.append({
                'time': tr['time'],
                'doa': tr['doa_smooth'],
                't_min': tr['time'].min(),
                't_max': tr['time'].max()
            })

        dt_candidates = []
        for d in track_data:
            t = d['time']
            if len(t) >= 2:
                med_dt = np.median(np.diff(t))
                if np.isfinite(med_dt) and med_dt > 0:
                    dt_candidates.append(med_dt)
        if not dt_candidates:
            return [], 0
        dt = min(dt_candidates)

        grid_start = min(d['t_min'] for d in track_data)
        grid_end = max(d['t_max'] for d in track_data)
        common_grid = np.arange(grid_start, grid_end, dt)
        n_times = len(common_grid)
        if n_times == 0:
            return [], 0

        doa_matrix = np.full((n_tracks, n_times), np.nan, dtype=np.float32)
        for i, d in enumerate(track_data):
            mask = (common_grid >= d['t_min']) & (common_grid <= d['t_max'])
            if mask.any():
                doa_matrix[i, mask] = np.interp(
                    common_grid[mask], d['time'], d['doa']
                ).astype(np.float32)

        valid_mask = ~np.isnan(doa_matrix)
        valid_int = valid_mask.astype(np.int16)
        overlap_counts = valid_int @ valid_int.T

        idx_arr = np.arange(n_tracks)
        upper_tri = idx_arr[:, None] < idx_arr[None, :]
        qualifies = (overlap_counts >= self.min_overlap_points) & upper_tri
        pairs_i, pairs_j = np.where(qualifies)

        if len(pairs_i) == 0:
            return [], 0

        pair_results = []
        consistent_count = 0

        for k in range(len(pairs_i)):
            i, j = pairs_i[k], pairs_j[k]
            overlap = valid_mask[i] & valid_mask[j]
            d1 = doa_matrix[i, overlap]
            d2 = doa_matrix[j, overlap]

            diffs = np.abs(circular_signed_diff_deg(d1, d2))
            mean_diff = float(np.mean(diffs))
            median_diff = float(np.median(diffs))

            consistent = (mean_diff <= self.doa_mean_thr) and (median_diff <= self.doa_median_thr)
            if consistent:
                consistent_count += 1

            pair_results.append({
                'tid1': tid_list[i],
                'tid2': tid_list[j],
                'consistent': consistent
            })

        return pair_results, consistent_count


# ============================================================
# 6. 集成流式处理器
# ============================================================

class StreamLineDOAClusterProcessor:
    """集成DOA估计和实时聚类的流式处理器"""
    def __init__(
        self,
        fs,
        model,
        device='cpu',
        process_window=80.0,
        process_hop=20.0,
        raw_buffer_seconds=100.0,
        stft_win_seconds=20.0,
        stft_hop_seconds=1.0,
        doa_delay=5.0,
        doa_win_len=10.0,
        doa_mode='center',
        frequency_resolution=20,
        f_lower_bound=0,
        f_higher_bound=100,
        denoise_thresh=0.1,
        line_channel=0,
        spec_freq_div=20.0,
        add_f_lower_bound=False,
        track_match_max_dt=15.0,
        track_match_max_df=1.0,
        analysis_window=600,
        debug_dir=None
    ):
        self.fs = fs
        self.model = model
        self.device = device
        self.process_window = float(process_window)
        self.process_hop = float(process_hop)
        self.raw_buffer_seconds = float(raw_buffer_seconds)
        self.stft_win_seconds = float(stft_win_seconds)
        self.stft_hop_seconds = float(stft_hop_seconds)
        self.stft_center_offset = self.stft_win_seconds / 2.0
        self.doa_delay = float(doa_delay)
        self.doa_win_len = float(doa_win_len)
        self.doa_mode = doa_mode
        self.frequency_resolution = frequency_resolution
        self.f_lower_bound = f_lower_bound
        self.f_higher_bound = f_higher_bound
        self.denoise_thresh = denoise_thresh
        self.line_channel = line_channel
        self.spec_freq_div = spec_freq_div
        self.add_f_lower_bound = add_f_lower_bound
        self.track_match_max_dt = track_match_max_dt
        self.track_match_max_df = track_match_max_df
        self.debug_dir = debug_dir

        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.pt_buffer = np.zeros(0, dtype=np.float32)
        self.vx_buffer = np.zeros(0, dtype=np.float32)
        self.vy_buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start_time = 0.0
        self.current_time = 0.0
        self.next_process_time = 80.0  # 第一次在80s处理

        self.pending_doa_points = []
        self.active_tracks = {}
        self.next_global_track_id = 1

        # 滑动全局std（EMA），用于稳定时频图归一化
        self.global_spec_std = None
        self.spec_std_ema_alpha = 0.1

        self.summary_noisy_images = []
        self.summary_denoise_images = []
        self.summary_trace_images = []
        self.summary_window_intervals = []

        # 累积总矩阵
        self.accumulated_noisy_spec = []
        self.accumulated_denoise_spec = []
        self.accumulated_doa_matrix = []
        self.accumulated_time_azimuth = []

        # 实时聚类分析器
        self.cluster_analyzer = RealtimeClusterAnalyzer(analysis_window=analysis_window)

        # 聚类结果输出文件
        self.cluster_output_file = None

        self.model.to(self.device)

    def push(self, pt_chunk, vx_chunk, vy_chunk):
        """推送数据并处理

        返回:
            dict: {
                'doa_results': list,  # DOA结果列表
                'noisy_spec': np.ndarray or None,  # 原始时频矩阵
                'denoise_spec': np.ndarray or None,  # 去噪后时频矩阵
                'doa_matrix': np.ndarray or None,  # DOA方位矩阵
                'time_azimuth': np.ndarray or None  # 时间-方位图
            }
        """
        import time
        t0 = time.time()

        pt_chunk = np.asarray(pt_chunk).flatten()
        vx_chunk = np.asarray(vx_chunk).flatten()
        vy_chunk = np.asarray(vy_chunk).flatten()

        if not (len(pt_chunk) == len(vx_chunk) == len(vy_chunk)):
            raise ValueError("pt/vx/vy chunk length mismatch")

        self._append_raw_data(pt_chunk, vx_chunk, vy_chunk)

        all_results = []
        window_specs = []  # 保存窗口的时频图
        last_window_end_time = None

        while self.current_time >= self.next_process_time:
            window_data = self._process_one_window(self.next_process_time)
            new_points = window_data['points']
            if len(new_points) > 0:
                self.pending_doa_points.extend(new_points)
            window_specs.append(window_data)
            last_window_end_time = self.next_process_time
            self.next_process_time += self.process_hop

        t_doa = time.time()
        doa_results = self._process_pending_doa()
        time_doa = time.time() - t_doa
        all_results.extend(doa_results)

        # 更新聚类分析器
        analysis_result = None
        if len(doa_results) > 0:
            self.cluster_analyzer.update_tracks(doa_results)

        # 每20秒执行一次聚类分析
        if len(doa_results) > 0:
            t_cluster = time.time()
            analysis_result = self.cluster_analyzer.analyze(self.current_time)
            time_cluster = time.time() - t_cluster

            elapsed = time.time() - t0
            print(f"[CLUSTER] time={self.current_time:.1f}s "
                  f"tracks={analysis_result['track_count']} "
                  f"valid={analysis_result.get('valid_track_count', 0)} "
                  f"clusters={analysis_result['cluster_count']} "
                  f"pairs={analysis_result['pair_count']}/{analysis_result['consistent_pair_count']} "
                  f"total={elapsed:.2f}s (DOA={time_doa:.2f}s cluster={time_cluster:.2f}s)")

            # 保存聚类结果
            if self.cluster_output_file is not None:
                self._save_cluster_result(analysis_result)

        # 构建返回的矩阵
        result_dict = {
            'doa_results': all_results,
            'noisy_spec': None,
            'denoise_spec': None,
            'doa_matrix': None,
            'time_azimuth': None,
            'analysis_result': analysis_result
        }

        # 如果有窗口数据，使用最新的窗口
        if len(window_specs) > 0:
            latest_window = window_specs[-1]
            noisy_full = latest_window.get('noisy_spec')
            denoise_full = latest_window.get('denoise_spec')

            # 只保留最新的process_hop秒数据
            if noisy_full is not None:
                # noisy_full原始格式: (时间, 频率)
                rows_to_keep = int(round(self.process_hop / self.stft_hop_seconds))
                # 裁剪时间维度（最后rows_to_keep行）
                result_dict['noisy_spec'] = noisy_full[-rows_to_keep:, :]
                result_dict['denoise_spec'] = denoise_full[-rows_to_keep:, :]

        # 构建DOA矩阵和时间-方位图
        if len(all_results) > 0 and result_dict['noisy_spec'] is not None and last_window_end_time is not None:
            # 使用最后处理的窗口结束时间
            window_start = last_window_end_time - self.process_hop - self.stft_center_offset
            result_dict['doa_matrix'] = self._build_doa_matrix(all_results, result_dict['noisy_spec'].shape, window_start)
            result_dict['time_azimuth'] = self._build_time_azimuth_plot(all_results, result_dict['denoise_spec'], window_start)

        # 累积总矩阵
        if result_dict['noisy_spec'] is not None:
            self.accumulated_noisy_spec.append(result_dict['noisy_spec'])
            self.accumulated_denoise_spec.append(result_dict['denoise_spec'])
        if result_dict['doa_matrix'] is not None:
            self.accumulated_doa_matrix.append(result_dict['doa_matrix'])
        if result_dict['time_azimuth'] is not None:
            self.accumulated_time_azimuth.append(result_dict['time_azimuth'])

        return result_dict

    def _build_doa_matrix(self, doa_results, spec_shape, window_start):
        """构建DOA方位矩阵 - 格式(时间, 频率)

        参数:
            window_start: 输出矩阵第0行对应的绝对时间
        """
        time_bins, freq_bins = spec_shape  # 第一维时间，第二维频率
        doa_matrix = np.full((time_bins, freq_bins), -5, dtype=np.float32)

        # 只处理当前批次的DOA结果
        for r in doa_results:
            t = r['time']
            f = r['freq']
            doa = r['doa']

            if doa < 0:  # 无效DOA
                continue

            # 转换为矩阵坐标
            t_relative = t - window_start
            t_idx = int(round(t_relative / self.stft_hop_seconds))

            if self.add_f_lower_bound:
                f_idx = int(round((f - self.f_lower_bound) * self.spec_freq_div))
            else:
                f_idx = int(round(f * self.spec_freq_div))

            if 0 <= t_idx < time_bins and 0 <= f_idx < freq_bins:
                doa_matrix[t_idx, f_idx] = doa

        return doa_matrix

    def _build_time_azimuth_plot(self, doa_results, noisy_spec, window_start, azimuth_bins=360):
        """构建时间-方位图（纵轴时间，横轴方位0-360度）- 仅当前批次

        参数:
            doa_results: DOA结果列表
            noisy_spec: 时频图矩阵 (时间, 频率)
            window_start: 时频图第0行对应的绝对时间
        """
        if len(doa_results) == 0:
            return np.zeros((1, azimuth_bins), dtype=np.float32)

        # 确定时间范围
        times = [r['time'] for r in doa_results]
        t_min = min(times)
        t_max = max(times)
        time_range = max(t_max - t_min, 1.0)

        # 时间分辨率：1秒
        time_bins = int(np.ceil(time_range)) + 1
        time_azimuth = np.zeros((time_bins, azimuth_bins), dtype=np.float32)

        spec_time_bins, spec_freq_bins = noisy_spec.shape

        for r in doa_results:
            doa = r['doa']
            if doa < 0:
                continue

            t = r['time']
            f = r['freq']

            # 获取时频图对应位置的幅度值
            t_relative = t - window_start
            spec_t_idx = int(round(t_relative / self.stft_hop_seconds))

            if self.add_f_lower_bound:
                spec_f_idx = int(round((f - self.f_lower_bound) * self.spec_freq_div))
            else:
                spec_f_idx = int(round(f * self.spec_freq_div))

            # 提取幅度值
            amplitude = 1.0  # 默认值
            if 0 <= spec_t_idx < spec_time_bins and 0 <= spec_f_idx < spec_freq_bins:
                amplitude = float(noisy_spec[spec_t_idx, spec_f_idx])

            # 累加到时间-方位图
            t_idx = int(round(t - t_min))
            az_idx = int(round(doa)) % azimuth_bins

            if 0 <= t_idx < time_bins and 0 <= az_idx < azimuth_bins:
                time_azimuth[t_idx, az_idx] += amplitude

        return time_azimuth

    def _append_raw_data(self, pt, vx, vy):
        """添加原始数据缓存"""
        self.pt_buffer = np.concatenate([self.pt_buffer, pt])
        self.vx_buffer = np.concatenate([self.vx_buffer, vx])
        self.vy_buffer = np.concatenate([self.vy_buffer, vy])
        self.current_time += len(pt) / self.fs

        max_samples = int(round(self.raw_buffer_seconds * self.fs))
        if len(self.pt_buffer) > max_samples:
            remove_samples = len(self.pt_buffer) - max_samples
            self.pt_buffer = self.pt_buffer[remove_samples:]
            self.vx_buffer = self.vx_buffer[remove_samples:]
            self.vy_buffer = self.vy_buffer[remove_samples:]
            self.buffer_start_time += remove_samples / self.fs

    def _process_one_window(self, window_end_time):
        """处理一个完整窗口"""
        import time
        t0 = time.time()

        # 累积窗口：从0开始到当前时间，最多600s
        window_start_time = max(0.0, window_end_time - self.process_window)

        if window_start_time < self.buffer_start_time:
            return []

        pt_win = self._get_buffer_segment(self.pt_buffer, window_start_time, window_end_time)
        if pt_win is None:
            return []

        actual_window_len = window_end_time - window_start_time
        expected_len = int(round(actual_window_len * self.fs))
        if len(pt_win) < expected_len * 0.99:
            return []

        # 1. 生成时频图
        t1 = time.time()
        noisy_spec = self._wave_to_spec_window(pt_win)
        time_stft = time.time() - t1

        # 2. 模型去噪（分块并行）
        t2 = time.time()
        denoise_spec, conf = test_deepdenoiser_stft_fragment_tiled(
            self.model, noisy_spec, device=self.device,
            thresh=self.denoise_thresh, line_channel=self.line_channel
        )
        time_denoise = time.time() - t2

        # 3. 线谱提取
        t3 = time.time()
        trace_img_vis, peaks_img, trajs = extract_multi_lines(
            denoise_spec, out_prefix='stream', prominence=5,
            min_length=20, delta_f=10, fre_range=10
        )
        time_extract = time.time() - t3

        # 4. 轨迹坐标转换
        abs_trajs = self._convert_trajs_to_absolute(trajs, window_start_time)

        # 5. 只输出新增的20秒
        output_start = window_end_time - self.process_hop - self.stft_center_offset
        output_end = window_end_time - self.stft_center_offset

        selected_points = self._select_points_from_trajs(abs_trajs, output_start, output_end)

        t4 = time.time()
        selected_points = self._assign_global_track_ids(selected_points)
        time_associate = time.time() - t4

        # 6. 保存图像
        self._append_stream_summary_window(
            noisy_spec, denoise_spec, trace_img_vis,
            window_start_time, window_end_time
        )

        # 7. 保存调试图像
        if self.debug_dir is not None:
            self._save_debug_window_image(noisy_spec, denoise_spec, trace_img_vis, window_end_time)

        elapsed = time.time() - t0
        print(f"[PROCESS] window=[{window_start_time:.1f},{window_end_time:.1f}) "
              f"points={len(selected_points)} total={elapsed:.2f}s "
              f"(STFT={time_stft:.2f}s denoise={time_denoise:.2f}s extract={time_extract:.2f}s associate={time_associate:.2f}s)")

        return {
            'points': selected_points,
            'noisy_spec': noisy_spec,
            'denoise_spec': denoise_spec
        }

    def _get_buffer_segment(self, buffer, start_time, end_time):
        """从缓存取时间片段"""
        if start_time < self.buffer_start_time:
            return None
        start_idx = int(round((start_time - self.buffer_start_time) * self.fs))
        end_idx = int(round((end_time - self.buffer_start_time) * self.fs))
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(buffer))
        if end_idx <= start_idx:
            return None
        return buffer[start_idx:end_idx]

    def _wave_to_spec_window(self, pt_win):
        """波形转时频图"""
        total_samples = len(pt_win)
        usable_samples = total_samples // self.fs * self.fs
        pt_win = pt_win[:usable_samples]
        wave = pt_win.reshape(-1, self.fs)

        spec = wave_to_spec(
            wave, frequency_resolution=self.frequency_resolution,
            fs=self.fs, f_lower_bound=self.f_lower_bound,
            f_higher_bound=self.f_higher_bound
        )

        spec = np.asarray(spec, dtype=np.float32)
        current_std = spec.std()
        if current_std > 1e-12:
            if self.global_spec_std is None:
                self.global_spec_std = current_std
            else:
                self.global_spec_std = (self.spec_std_ema_alpha * current_std +
                                         (1 - self.spec_std_ema_alpha) * self.global_spec_std)
            spec = spec / self.global_spec_std
        return spec

    def _convert_trajs_to_absolute(self, trajs, window_start_time):
        """轨迹坐标转绝对时间和频率"""
        abs_trajs = []
        for local_id, tra in enumerate(trajs):
            one_tra = []
            for point in tra:
                t_pix, f_pix = point
                abs_time = (window_start_time + self.stft_center_offset +
                           float(t_pix) * self.stft_hop_seconds)
                if self.add_f_lower_bound:
                    freq_hz = self.f_lower_bound + float(f_pix) / self.spec_freq_div
                else:
                    freq_hz = float(f_pix) / self.spec_freq_div
                one_tra.append({
                    "local_track_id": local_id + 1,
                    "time": abs_time,
                    "freq": freq_hz
                })
            abs_trajs.append(one_tra)
        return abs_trajs

    def _select_points_from_trajs(self, abs_trajs, output_start, output_end):
        """选择当前窗口内有效线谱点"""
        selected = []
        for tra in abs_trajs:
            for p in tra:
                t = p["time"]
                if output_start <= t < output_end:
                    selected.append({
                        "local_track_id": p["local_track_id"],
                        "time": p["time"],
                        "freq": p["freq"],
                        "global_track_id": None
                    })
        selected = sorted(selected, key=lambda x: (x["time"], x["freq"]))
        return selected

    def get_dynamic_freq_threshold(self, freq):
        """根据频率返回动态匹配阈值"""
        if freq <= 100:
            return 0.5
        elif freq >= 600:
            return 3.0
        else:
            # 100-600Hz之间线性插值
            return 0.5 + (freq - 100) / (600 - 100) * (3.0 - 0.5)

    def _assign_global_track_ids(self, selected_points):
        """分配全局轨迹ID"""
        if len(selected_points) == 0:
            return selected_points

        local_groups = {}
        for p in selected_points:
            lid = p["local_track_id"]
            if lid not in local_groups:
                local_groups[lid] = []
            local_groups[lid].append(p)

        for local_id, points in local_groups.items():
            points = sorted(points, key=lambda x: x["time"])
            first_point = points[0]
            t0 = first_point["time"]
            f0 = first_point["freq"]

            matched_gid = None
            best_score = 1e18

            dynamic_df_threshold = self.get_dynamic_freq_threshold(f0)

            for gid, track in self.active_tracks.items():
                last_t = track["last_time"]
                last_f = track["last_freq"]
                if last_t is None or last_f is None:
                    continue
                if not isinstance(last_t, (int, float)) or not isinstance(last_f, (int, float)):
                    continue
                dt = abs(t0 - last_t)
                df = abs(f0 - last_f)
                if dt <= self.track_match_max_dt and df <= dynamic_df_threshold:
                    score = df + 0.1 * dt
                    if score < best_score:
                        best_score = score
                        matched_gid = gid

            if matched_gid is None:
                matched_gid = self.next_global_track_id
                self.next_global_track_id += 1
                self.active_tracks[matched_gid] = {
                    "last_time": None,
                    "last_freq": None,
                    "points": []
                }

            for p in points:
                p["global_track_id"] = matched_gid
                self.active_tracks[matched_gid]["points"].append((p["time"], p["freq"]))
                self.active_tracks[matched_gid]["last_time"] = p["time"]
                self.active_tracks[matched_gid]["last_freq"] = p["freq"]

        return selected_points

    def _process_pending_doa(self):
        """DOA队列处理 - 批量优化版本"""
        import time
        t_start = time.time()

        results = []
        remain = []

        # 收集可处理的点
        points_to_process = []
        point_info = []

        for p in self.pending_doa_points:
            t = p["time"]
            f = p["freq"]
            if self.current_time < t + self.doa_delay:
                remain.append(p)
                continue

            points_to_process.append((t, f))
            point_info.append(p)

        # 批量处理DOA
        if len(points_to_process) > 0:
            doa_results = vector_doa_batch(
                self.pt_buffer, self.vx_buffer, self.vy_buffer, self.fs,
                self.buffer_start_time, points_to_process,
                win_len=self.doa_win_len, mode=self.doa_mode, device=self.device
            )

            for p, doa in zip(point_info, doa_results):
                if doa is None:
                    remain.append(p)
                    continue

                results.append({
                    "track_id": p["global_track_id"],
                    "time": p["time"],
                    "freq": p["freq"],
                    "doa": doa
                })

        self.pending_doa_points = remain

        elapsed = time.time() - t_start
        if len(points_to_process) > 0:
            avg_time = elapsed / len(points_to_process)
            print(f"  [DOA] processed={len(points_to_process)} points, total={elapsed:.3f}s, avg={avg_time*1000:.2f}ms/point")

        return results

    def _append_stream_summary_window(self, noisy_spec, denoise_spec, trace_img_vis,
                                      window_start_time, window_end_time):
        """保存当前流式窗口图像"""
        noisy_img = to_image(noisy_spec)
        denoise_img = to_image(denoise_spec)
        trace_img = to_image(trace_img_vis)

        base_h, base_w = noisy_img.shape[:2]

        if denoise_img.shape[1] != base_w:
            denoise_img = cv2.resize(denoise_img, (base_w, denoise_img.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        if trace_img.shape[1] != base_w:
            trace_img = cv2.resize(trace_img, (base_w, trace_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        if denoise_img.shape[0] != base_h:
            denoise_img = cv2.resize(denoise_img, (base_w, base_h),
                                    interpolation=cv2.INTER_NEAREST)
        if trace_img.shape[0] != base_h:
            trace_img = cv2.resize(trace_img, (base_w, base_h),
                                  interpolation=cv2.INTER_NEAREST)

        # 除第一个窗口外，只保留后20秒
        is_first_window = len(self.summary_noisy_images) == 0
        if not is_first_window:
            rows_to_keep = int(round(self.process_hop / self.stft_hop_seconds))
            noisy_img = noisy_img[-rows_to_keep:, :]
            denoise_img = denoise_img[-rows_to_keep:, :]
            trace_img = trace_img[-rows_to_keep:, :]

        self.summary_noisy_images.append(noisy_img)
        self.summary_denoise_images.append(denoise_img)
        self.summary_trace_images.append(trace_img)
        self.summary_window_intervals.append((window_start_time, window_end_time))

    def get_accumulated_matrices(self):
        """获取累积的总矩阵"""
        result = {
            'noisy_spec': None,
            'denoise_spec': None,
            'doa_matrix': None,
            'time_azimuth': None
        }

        if len(self.accumulated_noisy_spec) > 0:
            result['noisy_spec'] = np.vstack(self.accumulated_noisy_spec)
            result['denoise_spec'] = np.vstack(self.accumulated_denoise_spec)

        if len(self.accumulated_doa_matrix) > 0:
            result['doa_matrix'] = np.vstack(self.accumulated_doa_matrix)

        if len(self.accumulated_time_azimuth) > 0:
            result['time_azimuth'] = np.vstack(self.accumulated_time_azimuth)

        return result

    def save_accumulated_matrices_image(self, save_path):
        """保存累积总矩阵的可视化图像"""
        matrices = self.get_accumulated_matrices()

        if matrices['noisy_spec'] is None:
            print("[WARN] no accumulated matrices to save.")
            return

        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        # 转换为图像
        noisy_img = to_image(matrices['noisy_spec'])
        denoise_img = to_image(matrices['denoise_spec'])

        # DOA矩阵可视化 - 使用HSV色彩空间
        doa_img = None
        if matrices['doa_matrix'] is not None:
            doa_mat = matrices['doa_matrix'].copy()
            h, w = doa_mat.shape

            # 创建HSV图像
            hsv_img = np.zeros((h, w, 3), dtype=np.uint8)

            # 有效值掩码
            valid_mask = doa_mat >= 0

            # 将0-360度映射到0-179 (OpenCV的Hue范围)
            hsv_img[valid_mask, 0] = (doa_mat[valid_mask] / 360.0 * 179).astype(np.uint8)
            hsv_img[valid_mask, 1] = 255  # 饱和度
            hsv_img[valid_mask, 2] = 255  # 亮度

            # 转换为BGR
            doa_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # 统一宽度
        base_h, base_w = noisy_img.shape[:2]

        if denoise_img.shape[1] != base_w:
            denoise_img = cv2.resize(denoise_img, (base_w, denoise_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if denoise_img.shape[0] != base_h:
            denoise_img = cv2.resize(denoise_img, (base_w, base_h), interpolation=cv2.INTER_NEAREST)

        images_to_stack = [noisy_img, denoise_img]

        if doa_img is not None:
            if doa_img.shape[1] != base_w:
                doa_img = cv2.resize(doa_img, (base_w, doa_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            if doa_img.shape[0] != base_h:
                doa_img = cv2.resize(doa_img, (base_w, base_h), interpolation=cv2.INTER_NEAREST)
            images_to_stack.append(doa_img)

        # 添加频率刻度尺
        ruler = make_frequency_ruler(base_w, self.f_lower_bound, self.f_higher_bound, height=40)

        # 组合图像
        final_images = []
        for img in images_to_stack:
            final_images.append(ruler)
            final_images.append(img)
        final_images.append(ruler)

        all_image = np.vstack(final_images)
        cv2.imwrite(save_path, all_image)
        print(f"[INFO] accumulated matrices image saved to: {save_path}")

        # 时间-方位图单独保存
        if matrices['time_azimuth'] is not None:
            time_az_img = to_image(matrices['time_azimuth'])
            time_az_path = save_path.replace('.png', '_time_azimuth.png')
            cv2.imwrite(time_az_path, time_az_img)
            print(f"[INFO] time-azimuth image saved to: {time_az_path}")

        # 保存原始矩阵为TXT
        base_path = save_path.replace('.png', '')
        np.savetxt(f"{base_path}_noisy.txt", matrices['noisy_spec'], delimiter=',', fmt='%.6f')
        np.savetxt(f"{base_path}_denoise.txt", matrices['denoise_spec'], delimiter=',', fmt='%.6f')
        if matrices['doa_matrix'] is not None:
            np.savetxt(f"{base_path}_doa.txt", matrices['doa_matrix'], delimiter=',', fmt='%.6f')
        if matrices['time_azimuth'] is not None:
            np.savetxt(f"{base_path}_time_azimuth.txt", matrices['time_azimuth'], delimiter=',', fmt='%.6f')
        print(f"[INFO] matrices saved as TXT files")

    def save_stream_summary_image(self, save_path):
        """保存最终汇总图"""
        if len(self.summary_noisy_images) == 0:
            print("[WARN] no stream summary images to save.")
            return

        noisy_spec_image = np.vstack(self.summary_noisy_images)
        denoise_image = np.vstack(self.summary_denoise_images)
        trace_img_vis = np.vstack(self.summary_trace_images)

        width = noisy_spec_image.shape[1]
        ruler = make_frequency_ruler(width, self.f_lower_bound,
                                     self.f_higher_bound, height=40)

        all_image = np.vstack([ruler, noisy_spec_image, ruler,
                              denoise_image, ruler, trace_img_vis, ruler])

        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(save_path, all_image)
        print(f"[INFO] stream summary image saved to: {save_path}")

    def _save_cluster_result(self, analysis_result):
        """保存单次聚类结果"""
        with open(self.cluster_output_file, 'a', encoding='utf-8') as f:
            # 写入时间戳和统计信息
            f.write(f"{self.current_time:.1f},{analysis_result['track_count']},"
                   f"{analysis_result['valid_track_count']},{analysis_result['cluster_count']},"
                   f"{analysis_result['pair_count']},{analysis_result['consistent_pair_count']}\n")

            # 写入每个簇的详细信息
            for i, cluster in enumerate(analysis_result['clusters'], 1):
                # 只记录当前周期检出的线谱ID
                active_ids = [tid for tid in cluster
                              if self.cluster_analyzer.all_tracks[tid]['t_end']
                              >= self.current_time - self.process_hop]
                if len(active_ids) > 0:
                    track_ids = ','.join(map(str, active_ids))
                    f.write(f"  cluster_{i},{len(active_ids)},{track_ids}\n")

    def _save_debug_window_image(self, noisy_spec, denoise_spec, trace_img_vis, window_end_time):
        """保存单个窗口的调试图像"""
        noisy_img = to_image(noisy_spec)
        denoise_img = to_image(denoise_spec)
        trace_img = to_image(trace_img_vis)

        base_h, base_w = noisy_img.shape[:2]

        if denoise_img.shape[1] != base_w:
            denoise_img = cv2.resize(denoise_img, (base_w, denoise_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if trace_img.shape[1] != base_w:
            trace_img = cv2.resize(trace_img, (base_w, trace_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if denoise_img.shape[0] != base_h:
            denoise_img = cv2.resize(denoise_img, (base_w, base_h), interpolation=cv2.INTER_NEAREST)
        if trace_img.shape[0] != base_h:
            trace_img = cv2.resize(trace_img, (base_w, base_h), interpolation=cv2.INTER_NEAREST)

        ruler = make_frequency_ruler(base_w, self.f_lower_bound, self.f_higher_bound, height=40)
        debug_img = np.vstack([ruler, noisy_img, ruler, denoise_img, ruler, trace_img, ruler])

        debug_path = os.path.join(self.debug_dir, f"window_{window_end_time:.1f}s.png")
        cv2.imwrite(debug_path, debug_img)


# ============================================================
# 7. 主函数
# ============================================================

if __name__ == "__main__":
    # 配置参数
    device = "cpu"
    tcp_host = "0.0.0.0"  # 监听所有网卡
    tcp_port = 18888  # TCP服务器端口

    model_path = (
        "work_dir/"
        "yolo_4feat_deepnoise_log_randomh_613_moremorexianpu_snrr0.05/"
        "yolo_deepDenoiser_20_240.pth"
    )

    output_txt = f"stream_output_tcp/tcp_realtime_cluster_result.txt"

    # StreamLineDOAClusterProcessor 参数
    fs = 5000
    process_window = 180.0
    process_hop = 20.0
    raw_buffer_seconds = process_window + 20
    stft_win_seconds = 20.0
    stft_hop_seconds = 1.0
    doa_delay = 5.0
    doa_win_len = 10.0
    doa_mode = 'center'
    frequency_resolution = 20
    f_lower_bound = 0                         # 低频从0开始
    f_higher_bound = 1000                      # 高频段，可以设置100开始测试，最终是1000的
    denoise_thresh = 0.1
    line_channel = 0
    spec_freq_div = 20.0
    add_f_lower_bound = False
    track_match_max_dt = 240
    track_match_max_df = 1.0
    debug_dir = 'stream_debug_images'

    # RealtimeClusterAnalyzer 参数
    analysis_window = 600
    doa_outlier_win = 5
    doa_outlier_thr = 20.0
    doa_smooth_win = 5
    min_overlap_points = 120
    doa_mean_thr = 30.0
    doa_median_thr = 30.0
    min_abs_doa_slope = 0.01
    max_doa_disp = 25.0
    min_cluster_size = 3
    k_core = 2

    # 加载模型
    print(f"[INFO] Loading model from {model_path}...")
    model = YoloDenoiser(out_channel=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"[INFO] Model loaded successfully")

    # 输出文件准备
    output_dir = os.path.dirname(output_txt)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("track_id,time,freq,doa\n")

    # 聚类结果文件
    cluster_txt = output_txt.replace(".txt", "_cluster.txt")
    with open(cluster_txt, 'w', encoding='utf-8') as f:
        f.write("# Format: time,track_count,valid_count,cluster_count,pair_count,consistent_count\n")
        f.write("#   cluster_N,size,track_ids\n")

    print(f"[INFO] All initialization complete")

    # 初始化处理器
    print(f"[INFO] Initializing processor with fs={fs}...")
    processor = StreamLineDOAClusterProcessor(
        fs=fs,
        model=model,
        device=device,
        process_window=process_window,
        process_hop=process_hop,
        raw_buffer_seconds=raw_buffer_seconds,
        stft_win_seconds=stft_win_seconds,
        stft_hop_seconds=stft_hop_seconds,
        doa_delay=doa_delay,
        doa_win_len=doa_win_len,
        doa_mode=doa_mode,
        frequency_resolution=frequency_resolution,
        f_lower_bound=f_lower_bound,
        f_higher_bound=f_higher_bound,
        denoise_thresh=denoise_thresh,
        line_channel=line_channel,
        spec_freq_div=spec_freq_div,
        add_f_lower_bound=add_f_lower_bound,
        track_match_max_dt=track_match_max_dt,
        track_match_max_df=track_match_max_df,
        analysis_window=analysis_window,
        debug_dir=debug_dir
    )

    # 配置聚类分析器参数
    processor.cluster_analyzer.doa_outlier_win = doa_outlier_win
    processor.cluster_analyzer.doa_outlier_thr = doa_outlier_thr
    processor.cluster_analyzer.doa_smooth_win = doa_smooth_win
    processor.cluster_analyzer.min_overlap_points = min_overlap_points
    processor.cluster_analyzer.doa_mean_thr = doa_mean_thr
    processor.cluster_analyzer.doa_median_thr = doa_median_thr
    processor.cluster_analyzer.min_abs_doa_slope = min_abs_doa_slope
    processor.cluster_analyzer.max_doa_disp = max_doa_disp
    processor.cluster_analyzer.min_cluster_size = min_cluster_size
    processor.cluster_analyzer.k_core = k_core
    processor.cluster_output_file = cluster_txt

    print(f"[INFO] Processor initialized, ready to receive data")

    # 启动TCP服务器（所有初始化完成后）
    print(f"[INFO] Starting TCP server on {tcp_host}:{tcp_port}...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((tcp_host, tcp_port))
    server_sock.listen(1)
    print(f"[INFO] Waiting for client connection...")

    sock, client_addr = server_sock.accept()
    print(f"[INFO] Client connected from {client_addr}")

    all_doa_results = []

    # TCP实时接收循环
    import time
    total_start_time = time.time()
    sec = 0

    def _save_results(results):
        """保存单次 push 的结果"""
        if len(results['doa_results']) == 0:
            return
        all_doa_results.extend(results['doa_results'])

        # 创建时间戳文件夹
        current_time = processor.current_time
        time_dir = os.path.join(output_dir, f"t_{current_time:.0f}s")
        os.makedirs(time_dir, exist_ok=True)

        # 保存DOA结果
        with open(output_txt, 'a', encoding='utf-8') as f:
            for r in results['doa_results']:
                line = f'{int(r["track_id"])},{r["time"]:.2f},{r["freq"]:.2f},{r["doa"]:.2f}\n'
                f.write(line)

        # 保存矩阵
        if results['noisy_spec'] is not None:
            np.savetxt(f"{time_dir}/noisy.txt", results['noisy_spec'], delimiter=',', fmt='%.6f')
            np.savetxt(f"{time_dir}/denoise.txt", results['denoise_spec'], delimiter=',', fmt='%.6f')
        if results['doa_matrix'] is not None:
            np.savetxt(f"{time_dir}/doa.txt", results['doa_matrix'], delimiter=',', fmt='%.6f')
        if results['time_azimuth'] is not None:
            np.savetxt(f"{time_dir}/time_azimuth.txt", results['time_azimuth'], delimiter=',', fmt='%.6f')

        # 保存聚类簇
        if results['analysis_result'] is not None and results['analysis_result']['cluster_count'] > 0:
            cluster_file = f"{time_dir}/clusters.txt"
            with open(cluster_file, 'w', encoding='utf-8') as f:
                for cluster in results['analysis_result']['clusters']:
                    f.write(','.join(map(str, cluster)) + '\n')

    try:
        while True:
            fs_recv, pt_1s, vx_1s, vy_1s = receive_tcp_data(sock)
            results = processor.push(pt_1s, vx_1s, vy_1s)
            if sec % 20 == 0:
                print('sec', sec)
            _save_results(results)
            sec += 1

    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        sock.close()
        total_elapsed = time.time() - total_start_time

        # 保存汇总图
        summary_png = output_txt.replace(".txt", "_stream_summary.png")
        processor.save_stream_summary_image(summary_png)

        # 保存累积矩阵
        accumulated_png = output_txt.replace(".txt", "_accumulated_matrices.png")
        processor.save_accumulated_matrices_image(accumulated_png)

        print(f"[INFO] stream processing finished.")
        print(f"[INFO] total processing time: {total_elapsed:.2f}s")
        print(f"[INFO] result saved to: {output_txt}")
        print(f"[INFO] total doa results: {len(all_doa_results)}")
