import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile

from psignal.lofar import norm, spec_single_stft_log10
from testfile.extract_multi_lines import extract_multi_lines
from yolo_denoiser import YoloDenoiser


# ============================================================
# 1. 波形转时频图
# ============================================================

def wave_to_spec(wave, frequency_resolution, fs, f_lower_bound, f_higher_bound):
    """
    将波形转换为时频图。

    注意：
        你当前的时频图约定是：
            横轴 = 频率
            纵轴 = 时间
    """

    f_lower_bound = int(f_lower_bound)
    f_higher_bound = int(f_higher_bound)

    assert len(wave.shape) == 2

    wave = norm(wave)

    spec = spec_single_stft_log10(
        wave,
        f_lower_bound,
        f_higher_bound,
        frequency_resolution,
        fs
    )

    spec = norm(spec)

    return spec


# ============================================================
# 2. 谱图转图像
# ============================================================

def to_image(data):
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


# ============================================================
# 3. 频率刻度尺
# ============================================================

def make_frequency_ruler(
    width,
    f_lower_bound,
    f_higher_bound,
    height=40
):
    """
    生成频率刻度尺。

    因为你的时频图：
        横轴 = 频率
        纵轴 = 时间

    所以 ruler 应该画频率，而不是时间。
    """

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
        cv2.putText(
            ruler,
            text,
            (min(x + 2, width - 60), 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        f += major_tick

    return ruler


# ============================================================
# 4. 模型去噪
# ============================================================

def test_deepdenoiser_stft_fragment(
    model,
    noise_spec,
    device='cpu',
    thresh=0.1,
    line_channel=0
):
    """
    对单张时频图做深度模型去噪。

    noise_spec:
        shape = [H, W]
        这里按你的约定：
            H = 时间
            W = 频率
    """

    ori_h, ori_w = noise_spec.shape

    target_h = (ori_h - 1) // 32 + 1
    target_w = (ori_w - 1) // 32 + 1

    pad_h = target_h * 32
    pad_w = target_w * 32

    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:ori_h, :ori_w] = noise_spec.astype(np.float32)

    x = torch.tensor(padded, dtype=torch.float32)
    x = x.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    x = x.to(device)


    with torch.no_grad():
        print('x', x.shape)

        res = model(x, flag='test')

        sf = nn.Softmax(dim=1).to(device)
        res = sf(res)

        res = res.detach().cpu().numpy()[0]

        conf = res[line_channel]
        conf = conf[:ori_h, :ori_w]

        conf_soft = conf.copy()
        # conf_soft[conf_soft < thresh] = 0
        conf_soft[:, :2000] = np.where(conf_soft[:, :2000] < thresh, 0, conf_soft[:, :2000])
        conf_soft[:, 2000:] = np.where(conf_soft[:, 2000:] < 0.4, 0, conf_soft[:, 2000:])
        foreground = conf_soft * noise_spec

        conf_bin = conf_soft.copy()
        conf_bin[conf_bin < 0.4] = 0
        conf_bin[conf_bin >= 0.1] = 1
        conf_bin = conf_bin.astype(np.int32)

    return foreground, conf_bin


# ============================================================
# 5. 单点 DOA 估计
# ============================================================

def vector_doa_one_point(
    pt_buffer,
    vx_buffer,
    vy_buffer,
    fs,
    buffer_start_time,
    target_time,
    target_freq,
    win_len=10,
    mode='center'
):
    pt_buffer = np.asarray(pt_buffer, dtype=float).flatten()
    vx_buffer = np.asarray(vx_buffer, dtype=float).flatten()
    vy_buffer = np.asarray(vy_buffer, dtype=float).flatten()

    current_time = buffer_start_time + len(pt_buffer) / fs

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

    p = pt_buffer[start_idx:end_idx]
    x = vx_buffer[start_idx:end_idx]
    y = vy_buffer[start_idx:end_idx]

    def safe_normalize(sig):
        max_v = np.max(np.abs(sig))
        if max_v <= 1e-12:
            return sig.copy()
        return sig / max_v

    p = safe_normalize(p)
    x = safe_normalize(x)
    y = safe_normalize(y)

    sig_len = len(p)
    nfft = int(win_len * fs)

    win = np.hanning(sig_len)

    p_fft = np.fft.fft(p * win, n=nfft) / sig_len
    x_fft = np.fft.fft(x * win, n=nfft) / sig_len
    y_fft = np.fft.fft(y * win, n=nfft) / sig_len

    half_nfft = nfft // 2

    def matlab_round_pos(v):
        return int(np.floor(v + 0.5))

    f_low = max(target_freq - 0.5, 0)
    f_high = target_freq + 0.5

    f_ln = max(1, matlab_round_pos(f_low * nfft / fs))
    f_hn = min(half_nfft, matlab_round_pos(f_high * nfft / fs))

    if f_ln > f_hn:
        return None

    p_seg = p_fft[f_ln - 1:f_hn]
    x_seg = x_fft[f_ln - 1:f_hn]
    y_seg = y_fft[f_ln - 1:f_hn]

    if len(p_seg) == 0:
        return None

    Pvx2 = np.real(p_seg * np.conj(x_seg))
    Pvy2 = np.real(p_seg * np.conj(y_seg))

    est_angle = np.degrees(np.arctan2(Pvy2, Pvx2))

    theta_rad = np.radians(est_angle)
    theta_unwrap_rad = np.unwrap(theta_rad)
    est_angle = np.degrees(theta_unwrap_rad)

    est_angle = np.mod(est_angle + 180, 360)

    amp = np.abs(p_seg)

    if len(amp) == 0 or np.max(amp) <= 1e-12:
        return -5

    weight = amp / np.max(amp)

    nbins = 720
    theta_edges = np.linspace(0, 360, nbins + 1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0

    bin_idx = np.digitize(est_angle, theta_edges, right=False) - 1
    bin_idx[bin_idx == nbins] = nbins - 1
  
    valid = (
        np.isfinite(est_angle)
        & np.isfinite(weight)
        & (bin_idx >= 0)
        & (bin_idx < nbins)
    )

    if not np.any(valid):
        return -5

    count_weight = np.zeros(nbins, dtype=float)
    np.add.at(count_weight, bin_idx[valid], weight[valid])

    max_idx = np.argmax(count_weight)
    doa = theta_centers[max_idx]

    return doa


# ============================================================
# 6. 流式处理器
# ============================================================

class SlidingStreamLineDOAProcessor:
    def __init__(
        self,
        fs,
        model,
        device='cpu',

        # 这里改成 60s，满足你说的“拼接每60s的结果”
        process_window=60.0,
        process_hop=60.0,
        raw_buffer_seconds=80.0,

        stft_win_seconds=20.0,
        stft_hop_seconds=1.0,

        doa_delay=5.0,
        doa_win_len=10.0,
        doa_mode='center',

        frequency_resolution=20,
        f_lower_bound=0,
        f_higher_bound=400,

        denoise_thresh=0.1,
        line_channel=0,

        spec_freq_div=20.0,
        add_f_lower_bound=False,

        track_match_max_dt=15.0,
        track_match_max_df=1.0,

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

        self.next_process_time = float(process_window)

        self.pending_doa_points = []

        self.active_tracks = {}
        self.next_global_track_id = 1

        # ========================================================
        # 最终汇总图缓存
        #
        # 这里保存的是每一个流式窗口完整结果图。
        #
        # 注意：
        #   你的时频图是：
        #       横轴 = 频率
        #       纵轴 = 时间
        #
        # 所以最后汇总时必须 np.vstack，沿时间方向向下拼。
        # ========================================================

        self.summary_noisy_images = []
        self.summary_denoise_images = []
        self.summary_trace_images = []
        self.summary_window_intervals = []

        self.model.to(self.device)


    # ------------------------------------------------------------
    # 外部推流接口
    # ------------------------------------------------------------

    def push(self, pt_chunk, vx_chunk, vy_chunk):
        pt_chunk = np.asarray(pt_chunk).flatten()
        vx_chunk = np.asarray(vx_chunk).flatten()
        vy_chunk = np.asarray(vy_chunk).flatten()

        if not (len(pt_chunk) == len(vx_chunk) == len(vy_chunk)):
            raise ValueError("pt/vx/vy chunk length mismatch")

        self._append_raw_data(pt_chunk, vx_chunk, vy_chunk)

        all_results = []

        while self.current_time >= self.next_process_time:
            new_points = self._process_one_window(self.next_process_time)

            if len(new_points) > 0:
                self.pending_doa_points.extend(new_points)

            self.next_process_time += self.process_hop

        doa_results = self._process_pending_doa()
        all_results.extend(doa_results)

        return all_results

    # ------------------------------------------------------------
    # 添加原始数据缓存
    # ------------------------------------------------------------

    def _append_raw_data(self, pt, vx, vy):
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

    # ------------------------------------------------------------
    # 处理一个完整 60s 流式窗口
    # ------------------------------------------------------------

    def _process_one_window(self, window_end_time):
        window_start_time = window_end_time - self.process_window

        if window_start_time < self.buffer_start_time:
            print(
                f"[WARN] buffer not enough. "
                f"need [{window_start_time:.2f}, {window_end_time:.2f}], "
                f"buffer_start={self.buffer_start_time:.2f}"
            )
            return []

        pt_win = self._get_buffer_segment(
            self.pt_buffer,
            window_start_time,
            window_end_time
        )

        if pt_win is None:
            return []

        expected_len = int(round(self.process_window * self.fs))

        if len(pt_win) < expected_len * 0.99:
            print(
                f"[WARN] pt_win length not enough. "
                f"got={len(pt_win)}, expected={expected_len}"
            )
            return []

        # 1. 当前 60s 窗口生成 noisy spec
        noisy_spec = self._wave_to_spec_window(pt_win)

        # 2. 当前 60s 窗口模型去噪
        denoise_spec, conf = test_deepdenoiser_stft_fragment(
            self.model,
            noisy_spec,
            device=self.device,
            thresh=self.denoise_thresh,
            line_channel=self.line_channel
        )

        # 3. 当前 60s 窗口线谱提取
        trace_img_vis, peaks_img, trajs = extract_multi_lines(
            denoise_spec,
            out_prefix='stream',
            prominence=5,
            min_length=20,
            delta_f=10,
            fre_range=10
        )

        # 4. 当前窗口轨迹坐标转绝对时间
        abs_trajs = self._convert_trajs_to_absolute(
            trajs,
            window_start_time
        )

        # ========================================================
        # 只输出新增的 process_hop 时段，避免重叠窗口重复输出
        #
        # 当前窗口：[window_start_time, window_end_time)
        # 有效时频图：[window_start_time + 10, window_end_time - 10]
        #
        # 只输出最后 process_hop 秒的数据：
        #   output_start = window_end_time - process_hop - stft_center_offset
        #   output_end = window_end_time - stft_center_offset
        #
        # 例如：window=[0,80), hop=20, offset=10
        #   输出 [80-20-10, 80-10) = [50, 70)
        # ========================================================

        output_start = window_end_time - self.process_hop - self.stft_center_offset
        output_end = window_end_time - self.stft_center_offset

        selected_points = self._select_points_from_trajs(
            abs_trajs,
            output_start,
            output_end
        )

        selected_points = self._assign_global_track_ids(selected_points)

        # ========================================================
        # 关键：
        # 保存当前 60s 流式窗口的完整图像结果。
        # 不裁剪 10s。
        # 不横向拼接。
        # 最后按时间轴向下 vstack。
        # ========================================================

        self._append_stream_summary_window(
            noisy_spec=noisy_spec,
            denoise_spec=denoise_spec,
            trace_img_vis=trace_img_vis,
            window_start_time=window_start_time,
            window_end_time=window_end_time
        )

        if self.debug_dir is not None:
            self._save_debug_image(
                noisy_spec=noisy_spec,
                denoise_spec=denoise_spec,
                trace_img_vis=trace_img_vis,
                window_start_time=window_start_time,
                window_end_time=window_end_time,
                output_start=output_start,
                output_end=output_end
            )

        print(
            f"[PROCESS] raw_window=[{window_start_time:.1f},{window_end_time:.1f}) "
            f"valid_time=[{output_start:.1f},{output_end:.1f}) "
            f"points={len(selected_points)}"
        )

        return selected_points

    # ------------------------------------------------------------
    # 从缓存取时间片段
    # ------------------------------------------------------------

    def _get_buffer_segment(self, buffer, start_time, end_time):
        if start_time < self.buffer_start_time:
            return None

        start_idx = int(round((start_time - self.buffer_start_time) * self.fs))
        end_idx = int(round((end_time - self.buffer_start_time) * self.fs))

        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(buffer))

        if end_idx <= start_idx:
            return None

        return buffer[start_idx:end_idx]

    # ------------------------------------------------------------
    # 当前窗口波形转时频图
    # ------------------------------------------------------------

    def _wave_to_spec_window(self, pt_win):
        """
        当前代码沿用你原本写法：

            wave = pt_win.reshape(-1, self.fs)

        即每一行是 1s 数据。

        最终 spec 按你的约定：
            shape = [时间, 频率]
        """

        total_samples = len(pt_win)
        usable_samples = total_samples // self.fs * self.fs
        pt_win = pt_win[:usable_samples]

        wave = pt_win.reshape(-1, self.fs)

        print('self.frequency_resolution', self.frequency_resolution)
        print('wave.shape', wave.shape)

        spec = wave_to_spec(
            wave,
            frequency_resolution=self.frequency_resolution,
            fs=self.fs,
            f_lower_bound=self.f_lower_bound,
            f_higher_bound=self.f_higher_bound
        )

        spec = np.asarray(spec, dtype=np.float32)

        std = spec.std()
        if std > 1e-12:
            spec = spec / std

        return spec

    # ------------------------------------------------------------
    # 轨迹坐标转绝对时间和频率
    # ------------------------------------------------------------

    def _convert_trajs_to_absolute(self, trajs, window_start_time):
        """
        注意：
            你的图像约定：
                x = 频率方向
                y = 时间方向

        extract_multi_lines 返回 point 如果是：
            point = (t_pix, f_pix)
        那下面保持原逻辑。

        如果你实际返回的是：
            point = (x, y) = (freq_pix, time_pix)

        那就需要把这里改成：
            f_pix, t_pix = point

        你之前代码里用的是：
            t_pix, f_pix = point

        这里暂时保持你的原始约定。
        """

        abs_trajs = []

        for local_id, tra in enumerate(trajs):
            one_tra = []

            for point in tra:
                t_pix, f_pix = point

                abs_time = (
                    window_start_time
                    + self.stft_center_offset
                    + float(t_pix) * self.stft_hop_seconds
                )

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

    # ------------------------------------------------------------
    # 选择当前窗口内有效线谱点
    # ------------------------------------------------------------

    def _select_points_from_trajs(self, abs_trajs, output_start, output_end):
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

    # ------------------------------------------------------------
    # 分配全局轨迹 ID
    # ------------------------------------------------------------

    def _assign_global_track_ids(self, selected_points):
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

            for gid, track in self.active_tracks.items():
                last_t = track["last_time"]
                last_f = track["last_freq"]

                if last_t is None or last_f is None:
                    continue

                dt = abs(t0 - last_t)
                df = abs(f0 - last_f)

                if dt <= self.track_match_max_dt and df <= self.track_match_max_df:
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

                self.active_tracks[matched_gid]["points"].append(
                    (p["time"], p["freq"])
                )

                self.active_tracks[matched_gid]["last_time"] = p["time"]
                self.active_tracks[matched_gid]["last_freq"] = p["freq"]

        return selected_points

    # ------------------------------------------------------------
    # DOA 队列处理
    # ------------------------------------------------------------

    def _process_pending_doa(self):
        results = []
        remain = []

        for p in self.pending_doa_points:
            t = p["time"]
            f = p["freq"]

            if self.current_time < t + self.doa_delay:
                remain.append(p)
                continue

            doa = vector_doa_one_point(
                self.pt_buffer,
                self.vx_buffer,
                self.vy_buffer,
                self.fs,
                self.buffer_start_time,
                target_time=t,
                target_freq=f,
                win_len=self.doa_win_len,
                mode=self.doa_mode
            )

            if doa is None:
                remain.append(p)
                continue

            results.append({
                "track_id": p["global_track_id"],
                "time": t,
                "freq": f,
                "doa": doa
            })

        self.pending_doa_points = remain

        return results

    # ------------------------------------------------------------
    # 保存当前流式窗口完整图像，用于最终汇总
    # ------------------------------------------------------------

    def _append_stream_summary_window(
        self,
        noisy_spec,
        denoise_spec,
        trace_img_vis,
        window_start_time,
        window_end_time
    ):
        noisy_img = to_image(noisy_spec)
        denoise_img = to_image(denoise_spec)
        trace_img = to_image(trace_img_vis)

        # 统一宽度，也就是频率轴像素数必须一致
        base_h, base_w = noisy_img.shape[:2]

        if denoise_img.shape[1] != base_w:
            denoise_img = cv2.resize(
                denoise_img,
                (base_w, denoise_img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        if trace_img.shape[1] != base_w:
            trace_img = cv2.resize(
                trace_img,
                (base_w, trace_img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # 最好高度也一致。每个窗口都是 80s，理论上高度应该一致。
        # 如果 trace 或 denoise 高度不一致，这里统一到 noisy 的高度。
        if denoise_img.shape[0] != base_h:
            denoise_img = cv2.resize(
                denoise_img,
                (base_w, base_h),
                interpolation=cv2.INTER_NEAREST
            )

        if trace_img.shape[0] != base_h:
            trace_img = cv2.resize(
                trace_img,
                (base_w, base_h),
                interpolation=cv2.INTER_NEAREST
            )

        # 除第一个窗口外，只保留后 process_hop 秒对应的图像行
        # 纵轴是时间，每行对应 stft_hop_seconds
        is_first_window = len(self.summary_noisy_images) == 0

        if not is_first_window:
            # 计算需要保留的行数：process_hop / stft_hop_seconds
            rows_to_keep = int(round(self.process_hop / self.stft_hop_seconds))
            # 从底部（最新时间）保留
            noisy_img = noisy_img[-rows_to_keep:, :]
            denoise_img = denoise_img[-rows_to_keep:, :]
            trace_img = trace_img[-rows_to_keep:, :]

        self.summary_noisy_images.append(noisy_img)
        self.summary_denoise_images.append(denoise_img)
        self.summary_trace_images.append(trace_img)
        self.summary_window_intervals.append((window_start_time, window_end_time))

    # ------------------------------------------------------------
    # 保存最终流式汇总图
    # ------------------------------------------------------------

    def save_stream_summary_image(self, save_path):
        """
        保存最终汇总图。

        关键点：
            1. 用的是流式过程中每个 60s 窗口已经算出来的结果；
            2. 不会拿整段数据重新跑；
            3. 因为图像横轴是频率、纵轴是时间，所以窗口之间用 np.vstack；
            4. ruler 是频率尺，不是时间尺。
        """

        if len(self.summary_noisy_images) == 0:
            print("[WARN] no stream summary images to save.")
            return

        # ========================================================
        # 正确拼接方式：
        #
        # 横轴 = 频率
        # 纵轴 = 时间
        #
        # 所以按时间顺序向下拼：
        # ========================================================

        noisy_spec_image = np.vstack(self.summary_noisy_images)
        denoise_image = np.vstack(self.summary_denoise_images)
        trace_img_vis = np.vstack(self.summary_trace_images)

        width = noisy_spec_image.shape[1]

        ruler = make_frequency_ruler(
            width=width,
            f_lower_bound=self.f_lower_bound,
            f_higher_bound=self.f_higher_bound,
            height=40
        )

        all_image = np.vstack([
            ruler,
            noisy_spec_image,
            ruler,
            denoise_image,
            ruler,
            trace_img_vis,
            ruler
        ])

        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        ok = cv2.imwrite(save_path, all_image)

        if not ok:
            print(f"[ERROR] failed to save stream summary image: {save_path}")
        else:
            print(f"[INFO] stream summary image saved to: {save_path}")

        start_time = self.summary_window_intervals[0][0]
        end_time = self.summary_window_intervals[-1][1]

        print(f"[INFO] summary window count: {len(self.summary_window_intervals)}")
        print(f"[INFO] summary time range: [{start_time:.1f}, {end_time:.1f})")
        print(f"[INFO] summary image shape: {all_image.shape}")

    # ------------------------------------------------------------
    # 保存单窗口调试图
    # ------------------------------------------------------------

    def _save_debug_image(
        self,
        noisy_spec,
        denoise_spec,
        trace_img_vis,
        window_start_time,
        window_end_time,
        output_start,
        output_end
    ):
        noisy_img = to_image(noisy_spec)
        denoise_img = to_image(denoise_spec)
        trace_img = to_image(trace_img_vis)

        base_h, base_w = noisy_img.shape[:2]

        if denoise_img.shape[:2] != noisy_img.shape[:2]:
            denoise_img = cv2.resize(
                denoise_img,
                (base_w, base_h),
                interpolation=cv2.INTER_NEAREST
            )

        if trace_img.shape[:2] != noisy_img.shape[:2]:
            trace_img = cv2.resize(
                trace_img,
                (base_w, base_h),
                interpolation=cv2.INTER_NEAREST
            )

        # 因为纵轴是时间，所以有效时间边界应该画横线，而不是竖线。
        #
        # y = time
        #
        y1 = int(round(
            (output_start - window_start_time - self.stft_center_offset)
            / self.stft_hop_seconds
        ))

        y2 = int(round(
            (output_end - window_start_time - self.stft_center_offset)
            / self.stft_hop_seconds
        ))

        for img in [noisy_img, denoise_img, trace_img]:
            if 0 <= y1 < base_h:
                cv2.line(img, (0, y1), (base_w - 1, y1), (255, 255, 255), 1)
            if 0 <= y2 < base_h:
                cv2.line(img, (0, y2), (base_w - 1, y2), (255, 255, 255), 1)

        ruler = make_frequency_ruler(
            width=base_w,
            f_lower_bound=self.f_lower_bound,
            f_higher_bound=self.f_higher_bound,
            height=40
        )

        all_img = np.vstack([
            ruler,
            noisy_img,
            ruler,
            denoise_img,
            ruler,
            trace_img,
            ruler
        ])

        filename = f"window_{window_start_time:.0f}_{window_end_time:.0f}.png"
        save_path = os.path.join(self.debug_dir, filename)

        cv2.imwrite(save_path, all_img)


# ============================================================
# 7. 用 wav 文件模拟流式输入
# ============================================================

def run_stream_from_wav(
    ptData_file,
    vxData_file,
    vyData_file,
    model_path,
    output_txt,
    device='cpu'
):
    # ------------------------------------------------------------
    # 读取三通道数据
    # ------------------------------------------------------------

    fs_pt, ptData = wavfile.read(ptData_file)
    fs_vx, vxData = wavfile.read(vxData_file)
    fs_vy, vyData = wavfile.read(vyData_file)

    if not (fs_pt == fs_vx == fs_vy):
        raise ValueError("Pt/Vx/Vy sampling rate mismatch")

    fs = fs_pt

    min_len = min(len(ptData), len(vxData), len(vyData))

    ptData = ptData[:min_len]
    vxData = vxData[:min_len]
    vyData = vyData[:min_len]

    total_seconds = min_len // fs

    ptData = ptData[:total_seconds * fs]
    vxData = vxData[:total_seconds * fs]
    vyData = vyData[:total_seconds * fs]

    print(f"[INFO] fs={fs}, total_seconds={total_seconds}")

    # ------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------

    model = YoloDenoiser(out_channel=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # ------------------------------------------------------------
    # 初始化流式处理器
    #
    # 关键：
    #   process_window=60.0
    #   process_hop=60.0
    #
    # 表示：
    #   [0,60)
    #   [60,120)
    #   [120,180)
    #   ...
    #
    # 每次处理完整 60s，并把该 60s 图像结果存起来。
    # 最后按时间方向 vstack。
    # ------------------------------------------------------------

    processor = SlidingStreamLineDOAProcessor(
        fs=fs,
        model=model,
        device=device,

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

        debug_dir='stream_debug_images'
    )

    # ------------------------------------------------------------
    # 输出文件
    # ------------------------------------------------------------

    output_dir = os.path.dirname(output_txt)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("track_id,time,freq,doa\n")

    all_doa_results = []

    # ------------------------------------------------------------
    # 模拟每秒输入一次
    # ------------------------------------------------------------

    step = fs

    for sec in range(total_seconds):
        start = sec * step
        end = start + step

        pt_1s = ptData[start:end]
        vx_1s = vxData[start:end]
        vy_1s = vyData[start:end]

        results = processor.push(pt_1s, vx_1s, vy_1s)

        if len(results) > 0:
            all_doa_results.extend(results)

            with open(output_txt, 'a', encoding='utf-8') as f:
                for r in results:
                    line = (
                        f'{int(r["track_id"])},'
                        f'{r["time"]:.2f},'
                        f'{r["freq"]:.2f},'
                        f'{r["doa"]:.2f}\n'
                    )

                    f.write(line)

                    print(
                        "[DOA]",
                        "track_id=", r["track_id"],
                        "time=", f'{r["time"]:.2f}',
                        "freq=", f'{r["freq"]:.2f}',
                        "doa=", f'{r["doa"]:.2f}'
                    )

    # ------------------------------------------------------------
    # 最后保存流式汇总图
    #
    # 重点：
    #   这里不会用整段数据重新跑。
    #   只会把每个 60s 流式窗口结果沿时间方向 vstack。
    # ------------------------------------------------------------

    summary_png = output_txt.replace(".txt", "_stream_summary.png")

    processor.save_stream_summary_image(summary_png)

    print("[INFO] stream processing finished.")
    print(f"[INFO] result saved to: {output_txt}")
    print(f"[INFO] stream summary saved to: {summary_png}")
    print(f"[INFO] total doa results: {len(all_doa_results)}")


# ============================================================
# 8. 主入口
# ============================================================

if __name__ == "__main__":

    device = "cpu"
    # device = "cuda:0"

    pid = "20221128_143237"

    data_root = f"G:/jiuzhou/VectorDOA/jiuzhou_613data/{pid}"

    ptData_file = os.path.join(data_root, f"{pid}_Pt.wav")
    vxData_file = os.path.join(data_root, f"{pid}_Vx.wav")
    vyData_file = os.path.join(data_root, f"{pid}_Vy.wav")

    model_path = (
        "work_dir/"
        "yolo_4feat_deepnoise_log_randomh_613_moremorexianpu_snrr0.05/"
        "yolo_deepDenoiser_20_240.pth"
    )

    output_txt = f"stream_output/{pid}_stream_doa_result.txt"

    run_stream_from_wav(
        ptData_file=ptData_file,
        vxData_file=vxData_file,
        vyData_file=vyData_file,
        model_path=model_path,
        output_txt=output_txt,
        device=device
    )