import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import cv2
from scipy.io import wavfile
from psignal.lofar import norm, spec_single_stft_log10
from testfile.extract_multi_lines import extract_multi_lines


def wave_to_spec_log10(wave, frequency_resolution, fs, f_lower_bound, f_higher_bound):
    """Convert signal to time-spec image"""
    f_lower_bound = int(f_lower_bound)
    f_higher_bound = int(f_higher_bound)
    assert len(wave.shape) == 2
    wave = norm(wave)
    spec = spec_single_stft_log10(wave, f_lower_bound, f_higher_bound, frequency_resolution, fs)
    spec = norm(spec)
    return spec


def test_deepdenoiser_stft_fragment(model, noiseSpec, device='cpu', thresh=0.01):
    ori_h, ori_w = noiseSpec.shape
    target_h, target_w = (ori_h - 1) // 32 + 1, (ori_w - 1) // 32 + 1
    new_noiseSpec = np.zeros((target_h * 32, target_w * 32), dtype=np.float32)
    new_noiseSpec[:ori_h, :ori_w] = noiseSpec.astype(np.float32)
    new_noiseSpec = torch.tensor(new_noiseSpec).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    new_noiseSpec = new_noiseSpec.float()

    with torch.no_grad():
        res = model(new_noiseSpec, flag='train')
        sf = nn.Softmax(dim=1).to(device)
        res = sf(res)
        res = res.detach().cpu().numpy()[0]
        conf = res[0]
        conf = conf[:ori_h, :ori_w]
        conf[:, :2000] = np.where(conf[:, :2000] < thresh, 0, conf[:, :2000])
        conf[:, 2000:] = np.where(conf[:, 2000:] < 0.4, 0, conf[:, 2000:])

    foreground = conf * noiseSpec
    conf[conf >= thresh] = 1
    conf = conf.astype(np.int32)

    return foreground, conf


def vector_doa(ptData, vxData, vyData, fs, lineRecords):
    """Python 版本 VectorDOA"""
    ptData = np.asarray(ptData, dtype=float).flatten()
    vxData = np.asarray(vxData, dtype=float).flatten()
    vyData = np.asarray(vyData, dtype=float).flatten()
    lineRecords = np.asarray(lineRecords, dtype=float)

    if lineRecords.size == 0:
        return lineRecords

    if lineRecords.ndim == 1:
        lineRecords = lineRecords.reshape(1, -1)

    if lineRecords.shape[1] < 4:
        tmp = np.zeros((lineRecords.shape[0], 4), dtype=float)
        tmp[:, :lineRecords.shape[1]] = lineRecords
        lineRecords = tmp

    fs = int(fs)

    def matlab_round_pos(x):
        return np.floor(np.asarray(x) + 0.5).astype(int)

    T = len(ptData) / fs
    win_Len = 10
    nbins = 360 * 2
    Nfft = win_Len * fs

    theta = np.linspace(0, 360, nbins + 1)
    bin_centers = (theta[:-1] + theta[1:]) / 2

    def safe_normalize(x):
        m = np.max(np.abs(x))
        if m == 0:
            return x.copy()
        return x / m

    pdata = safe_normalize(ptData)
    vxdata = safe_normalize(vxData)
    vydata = safe_normalize(vyData)

    ids = lineRecords[:, 0]
    unique_ids = []
    groups = {}

    for idx, id_val in enumerate(ids):
        if id_val not in groups:
            groups[id_val] = []
            unique_ids.append(id_val)
        groups[id_val].append(idx)

    for id_val in unique_ids:
        row_indices = np.array(groups[id_val], dtype=int)
        tar_tf = lineRecords[row_indices, 1:3]
        M = tar_tf.shape[0]

        tar_time_center = tar_tf[:, 0]
        tar_time = np.column_stack([
            tar_time_center - 5,
            tar_time_center + 5
        ])

        tar_time[:, 0] = np.maximum(tar_time[:, 0], 1)
        tar_time[:, 1] = np.minimum(tar_time[:, 1], T)

        tar_freq_center = tar_tf[:, 1]
        tar_freq = np.column_stack([
            tar_freq_center - 0.5,
            tar_freq_center + 0.5
        ])

        tar_freq[:, 0] = np.maximum(tar_freq[:, 0], 10)

        fLn_array = np.maximum(
            1,
            matlab_round_pos(tar_freq[:, 0] * Nfft / fs)
        )

        fHn_array = np.minimum(
            Nfft // 2,
            matlab_round_pos(tar_freq[:, 1] * Nfft / fs)
        )

        for n in range(M):
            t_range = tar_time[n, :]

            start_idx = int(np.floor(t_range[0] * fs))
            end_idx = int(np.floor(t_range[1] * fs))

            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, len(pdata))

            if end_idx <= start_idx:
                continue

            if end_idx - start_idx < win_Len * fs:
                continue

            p = pdata[start_idx:end_idx]
            x = vxdata[start_idx:end_idx]
            y = vydata[start_idx:end_idx]

            sig_len = len(p)
            Nfft_local = min(Nfft, sig_len)

            win = np.hanning(sig_len)

            p_fft = np.fft.fft(p * win, n=Nfft_local) / sig_len
            x_fft = np.fft.fft(x * win, n=Nfft_local) / sig_len
            y_fft = np.fft.fft(y * win, n=Nfft_local) / sig_len

            half_nfft = Nfft_local // 2

            fLn = min(fLn_array[n], half_nfft)
            fHn = min(fHn_array[n], half_nfft)

            if fLn > fHn:
                continue

            p_seg = p_fft[fLn - 1:fHn]
            x_seg = x_fft[fLn - 1:fHn]
            y_seg = y_fft[fLn - 1:fHn]

            if len(p_seg) == 0:
                continue

            Pvx2 = np.real(p_seg * np.conj(x_seg))
            Pvy2 = np.real(p_seg * np.conj(y_seg))

            est_angle = np.degrees(np.arctan2(Pvy2, Pvx2))

            theta_rad = np.radians(est_angle)
            theta_unwrap_rad = np.unwrap(theta_rad)
            est_angle = np.degrees(theta_unwrap_rad)

            est_angle = np.mod(est_angle + 180, 360)

            Af = np.abs(p_seg)

            if np.all(Af == 0):
                theta_weight = -5
            else:
                max_Af = np.max(Af)

                if max_Af == 0 or not np.isfinite(max_Af):
                    theta_weight = -5
                else:
                    AddN = Af / max_Af

                    bin_idx = np.digitize(est_angle, theta, right=False) - 1
                    bin_idx[bin_idx == nbins] = nbins - 1

                    valid = (
                        np.isfinite(est_angle)
                        & np.isfinite(AddN)
                        & (bin_idx >= 0)
                        & (bin_idx < nbins)
                    )

                    if np.any(valid):
                        count_weight = np.zeros(nbins, dtype=float)

                        np.add.at(
                            count_weight,
                            bin_idx[valid],
                            AddN[valid]
                        )

                        max_idx_weight = np.argmax(count_weight)
                        theta_weight = bin_centers[max_idx_weight]
                    else:
                        theta_weight = -5

            lineRecords[row_indices[n], 3] = theta_weight

    return lineRecords


def process_audio_to_doa(audio_path, model, device='cpu', thresh=0.1):
    """
    完整的音频处理到DOA估计流程

    参数:
        audio_path: Pt.wav文件路径
        model: 降噪模型
        device: 设备
        thresh: 降噪阈值

    返回:
        new_lineRecords: 包含DOA角度的线谱记录 [ID, time, freq, doa_angle]
    """
    # 1. 加载音频
    audio_data, Fs = librosa.load(audio_path, sr=None)

    # 2. 截取360秒数据并reshape
    T = audio_data.shape[0] // Fs
    audio_data_split = audio_data[:360 * Fs]
    audio_data_split = audio_data_split.reshape(-1, Fs)

    # 3. 转换为频谱
    spec_data = wave_to_spec_log10(audio_data_split, frequency_resolution=20,
                                    fs=Fs, f_lower_bound=0, f_higher_bound=100)
    spec_data = spec_data[:, :-1]

    # 4. 标准化并截取前2000列
    noisy_spec = spec_data / spec_data.std()
    noisy_spec = noisy_spec[:, :2000]

    # 5. 降噪
    denoise_noisy_spec, conf = test_deepdenoiser_stft_fragment(model, noisy_spec, device, thresh)

    # 6. 提取线谱轨迹
    trace_img_vis, peaks_img, trajs = extract_multi_lines(
        denoise_noisy_spec, out_prefix='demo_multi',
        prominence=5, min_length=40, delta_f=10, fre_range=10
    )

    # 7. 转换轨迹数据格式
    trace_data = []
    for idx in range(len(trajs)):
        tra = trajs[idx]
        for j in range(len(tra)):
            t, f = tra[j]
            t += 10
            f = f / 20
            info = [idx+1, t, f]
            trace_data.append(info)
    trace_data = np.array(trace_data)

    # 8. 加载三通道数据
    pid = audio_path.split('/')[-1].split('_Pt')[0]
    data_root = f'G:/jiuzhou/VectorDOA/jiuzhou_613data/{pid}'

    fs, ptData = wavfile.read(f'{data_root}/{pid}_Pt.wav')
    fs, vxData = wavfile.read(f'{data_root}/{pid}_Vx.wav')
    fs, vyData = wavfile.read(f'{data_root}/{pid}_Vy.wav')

    # 9. 截取相同时长
    T = len(ptData) // fs
    ptData = ptData[:T * fs]
    vxData = vxData[:T * fs]
    vyData = vyData[:T * fs]

    # 10. DOA估计
    new_lineRecords = vector_doa(ptData, vxData, vyData, fs, trace_data)

    return new_lineRecords, noisy_spec, denoise_noisy_spec, trace_img_vis

