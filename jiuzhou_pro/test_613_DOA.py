import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ultralytics import YOLO
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
from scipy import io
from psignal.lofar import draw, norm, spec_single_stft, spec_single_stft_log10
from models.deepnoise_unet import DeepDenoiser
import cv2
from testfile.extract_multi_lines import extract_multi_lines
from scipy.io import wavfile
# from doa_python import vector_doa

def make_freq_ruler(width, height=45, px_per_hz=20, tick_hz=5):
    """
    生成横轴频率刻度尺
    
    width: 图像宽度
    height: 刻度尺高度
    px_per_hz: 多少像素代表 1 Hz
    tick_hz: 每多少 Hz 画一个刻度
    """
    ruler = np.ones((height, width, 3), dtype=np.uint8) * 255
    tick_px = px_per_hz * tick_hz  # 5Hz 对应 100 像素
    # 横线
    y_axis = 8
    cv2.line(ruler, (0, y_axis), (width - 1, y_axis), (0, 0, 0), 1)
    for x in range(0, width, tick_px):
        freq = x / px_per_hz
        # 主刻度线
        cv2.line(
            ruler,
            (x, y_axis),
            (x, y_axis + 12),
            (0, 0, 0),
            1
        )
        # 频率文字
        label = f"{int(freq)}Hz"
        cv2.putText(
            ruler,
            label,
            (x + 2, y_axis + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    return ruler
def vector_doa(ptData, vxData, vyData, fs, lineRecords):
    """
    Python 版本 VectorDOA，尽量对齐 MATLAB 版本。

    参数
    ----
    ptData : array_like
        声压信号数据向量

    vxData : array_like
        X 轴方向振速信号数据向量

    vyData : array_like
        Y 轴方向振速信号数据向量

    fs : float or int
        采样频率

    lineRecords : ndarray
        线谱信息矩阵
        第 1 列: 目标 ID
        第 2 列: 线谱中心时刻 time
        第 3 列: 线谱中心频率 freq

    返回
    ----
    lineRecords : ndarray
        更新后的线谱矩阵
        第 4 列为估计得到的 DOA 角度，单位：度
        如果估计失败，第 4 列保持原值，或者在无有效能量时赋值为 -5
    """

    # =========================
    # 输入转换
    # =========================
    ptData = np.asarray(ptData, dtype=float).flatten()
    vxData = np.asarray(vxData, dtype=float).flatten()
    vyData = np.asarray(vyData, dtype=float).flatten()
    lineRecords = np.asarray(lineRecords, dtype=float)

    if lineRecords.size == 0:
        return lineRecords

    if lineRecords.ndim == 1:
        lineRecords = lineRecords.reshape(1, -1)

    # 如果 lineRecords 只有 3 列，则扩展第 4 列
    if lineRecords.shape[1] < 4:
        tmp = np.zeros((lineRecords.shape[0], 4), dtype=float)
        tmp[:, :lineRecords.shape[1]] = lineRecords
        lineRecords = tmp

    fs = int(fs)

    # =========================
    # MATLAB 风格 round
    # MATLAB round(2.5)=3, round(3.5)=4
    # numpy round(2.5)=2, round(3.5)=4
    # 这里频率索引都是正数，所以 floor(x + 0.5) 即可
    # =========================
    def matlab_round_pos(x):
        return np.floor(np.asarray(x) + 0.5).astype(int)

    # 如果以后需要处理负数，可用这个：
    # def matlab_round(x):
    #     x = np.asarray(x)
    #     return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(int)

    # =========================
    # 基本参数
    # =========================
    T = len(ptData) / fs

    win_Len = 10
    nbins = 360 * 2
    Nfft = win_Len * fs

    theta = np.linspace(0, 360, nbins + 1)
    bin_centers = (theta[:-1] + theta[1:]) / 2

    # =========================
    # 信号归一化
    # =========================
    def safe_normalize(x):
        m = np.max(np.abs(x))
        if m == 0:
            return x.copy()
        return x / m

    pdata = safe_normalize(ptData)
    vxdata = safe_normalize(vxData)
    vydata = safe_normalize(vyData)

    # =========================
    # 按目标 ID 分组，保持 MATLAB unique(..., 'stable') 的顺序
    # =========================
    ids = lineRecords[:, 0]

    unique_ids = []
    groups = {}

    for idx, id_val in enumerate(ids):
        if id_val not in groups:
            groups[id_val] = []
            unique_ids.append(id_val)
        groups[id_val].append(idx)

    # =========================
    # 针对每个目标 ID 进行 DOA 估计
    # =========================
    for id_val in unique_ids:
        row_indices = np.array(groups[id_val], dtype=int)

        # 当前目标对应的 [time, freq]
        tar_tf = lineRecords[row_indices, 1:3]
        M = tar_tf.shape[0]

        # =========================
        # 时间窗口：[t - 5, t + 5]
        # 对齐 MATLAB:
        # tar_time(:,1) = max(tar_time(:,1), 1);
        # tar_time(:,2) = min(tar_time(:,2), T);
        # =========================
        tar_time_center = tar_tf[:, 0]
        tar_time = np.column_stack([
            tar_time_center - 5,
            tar_time_center + 5
        ])

        tar_time[:, 0] = np.maximum(tar_time[:, 0], 1)
        tar_time[:, 1] = np.minimum(tar_time[:, 1], T)

        # =========================
        # 频率带：freq ± 0.5 Hz，下限 10 Hz
        # =========================
        tar_freq_center = tar_tf[:, 1]
        tar_freq = np.column_stack([
            tar_freq_center - 0.5,
            tar_freq_center + 0.5
        ])

        tar_freq[:, 0] = np.maximum(tar_freq[:, 0], 10)

        # MATLAB:
        # fLn_array = max(1, round(tar_freq(:,1) * Nfft / fs));
        # fHn_array = min(Nfft/2, round(tar_freq(:,2) * Nfft / fs));
        #
        # 注意：这里使用 MATLAB 风格 round
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

            # MATLAB:
            # t_idx = floor(t_range(1)*fs) + 1 : floor(t_range(2)*fs);
            #
            # Python 0-based 等价：
            # MATLAB 取 [floor(t1*fs)+1, ..., floor(t2*fs)]
            # Python 下标为 [floor(t1*fs), ..., floor(t2*fs)-1]
            start_idx = int(np.floor(t_range[0] * fs))
            end_idx = int(np.floor(t_range[1] * fs))

            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, len(pdata))

            if end_idx <= start_idx:
                continue

            if end_idx - start_idx < win_Len * fs:
                # 与 MATLAB 一致：不足一个窗长就跳过
                continue

            p = pdata[start_idx:end_idx]
            x = vxdata[start_idx:end_idx]
            y = vydata[start_idx:end_idx]

            sig_len = len(p)
            Nfft_local = min(Nfft, sig_len)

            # MATLAB hanning(sig_len) 默认一般对应 symmetric Hann
            # np.hanning 与其基本一致
            win = np.hanning(sig_len)

            p_fft = np.fft.fft(p * win, n=Nfft_local) / sig_len
            x_fft = np.fft.fft(x * win, n=Nfft_local) / sig_len
            y_fft = np.fft.fft(y * win, n=Nfft_local) / sig_len

            half_nfft = Nfft_local // 2

            fLn = min(fLn_array[n], half_nfft)
            fHn = min(fHn_array[n], half_nfft)

            if fLn > fHn:
                continue

            # MATLAB:
            # p_seg = p_fft(fLn:fHn);
            #
            # Python 0-based：
            # fLn:fHn inclusive -> [fLn-1 : fHn]
            p_seg = p_fft[fLn - 1:fHn]
            x_seg = x_fft[fLn - 1:fHn]
            y_seg = y_fft[fLn - 1:fHn]

            if len(p_seg) == 0:
                continue

            # =========================
            # 交叉谱用于 DOA
            # MATLAB:
            # Pvx2 = real(p_seg .* conj(x_seg));
            # Pvy2 = real(p_seg .* conj(y_seg));
            # =========================
            Pvx2 = np.real(p_seg * np.conj(x_seg))
            Pvy2 = np.real(p_seg * np.conj(y_seg))

            est_angle = np.degrees(np.arctan2(Pvy2, Pvx2))

            theta_rad = np.radians(est_angle)
            theta_unwrap_rad = np.unwrap(theta_rad)
            est_angle = np.degrees(theta_unwrap_rad)

            est_angle = np.mod(est_angle + 180, 360)

            # =========================
            # 按幅度加权
            # =========================
            Af = np.abs(p_seg)

            if np.all(Af == 0):
                theta_weight = -5
            else:
                max_Af = np.max(Af)

                if max_Af == 0 or not np.isfinite(max_Af):
                    theta_weight = -5
                else:
                    AddN = Af / max_Af

                    # MATLAB:
                    # bin_idx = discretize(est_angle, theta);
                    #
                    # discretize 默认区间近似为：
                    # [theta(i), theta(i+1))
                    # 最后一个 bin 包含右边界。
                    #
                    # np.digitize 默认 right=False:
                    # bins[i-1] <= x < bins[i]
                    #
                    # 这里转换成 0-based bin index。
                    bin_idx = np.digitize(est_angle, theta, right=False) - 1

                    # 处理刚好等于 360 的情况
                    # 虽然 est_angle = mod(...,360) 后通常不会等于 360，
                    # 但为了贴近 MATLAB discretize，保留该边界处理。
                    bin_idx[bin_idx == nbins] = nbins - 1

                    valid = (
                        np.isfinite(est_angle)
                        & np.isfinite(AddN)
                        & (bin_idx >= 0)
                        & (bin_idx < nbins)
                    )

                    if np.any(valid):
                        count_weight = np.zeros(nbins, dtype=float)

                        # 等价 MATLAB accumarray
                        np.add.at(
                            count_weight,
                            bin_idx[valid],
                            AddN[valid]
                        )

                        max_idx_weight = np.argmax(count_weight)
                        theta_weight = bin_centers[max_idx_weight]
                    else:
                        theta_weight = -5

            # 写回 lineRecords 第 4 列
            lineRecords[row_indices[n], 3] = theta_weight

    return lineRecords

def wave_to_spec(wave, frequency_resolution, fs, f_lower_bound, f_higher_bound):
    """Convert signal to time-spec image

    Args:
        wave (np.array): original wave
        f_lower_bound (int):  lower bound of frequence range
        f_higher_bound (int): higher bound of frequence range

    Returns:
        np.array with shape (time_span, f_higher_bound-f_lower_bound)
    """
    f_lower_bound = int(f_lower_bound)
    f_higher_bound = int(f_higher_bound)
    assert len(wave.shape) == 2
    wave = norm(wave)
    spec = spec_single_stft_log10(wave, f_lower_bound, f_higher_bound, frequency_resolution, fs)
    spec = norm(spec)
    return spec

def to_image(data):
    # p_low = np.percentile(data, 5)
    # p_high = np.percentile(data, 95)
    # # 数据到5%-95%范围
    # data_clipped = np.clip(data, p_low, p_high)
    data_clipped = data.copy()
    image = ((data_clipped - data_clipped.min()) * 255 / (data_clipped.max() - data_clipped.min())).astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image

def test_deepdenoiser_stft_fragment(model, noiseSpec, thresh=0.01):
    ori_h, ori_w = noiseSpec.shape
    target_h, target_w = (ori_h - 1) // 32 + 1, (ori_w - 1) // 32 + 1
    new_noiseSpec = np.zeros((target_h * 32, target_w * 32), dtype=np.float32)
    new_noiseSpec[:ori_h, :ori_w] = noiseSpec.astype(np.float32)
    new_noiseSpec = torch.tensor(new_noiseSpec).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)  # → [3, H, W]
    new_noiseSpec = new_noiseSpec.float()
    print('new_noiseSpec', new_noiseSpec.shape)
    with torch.no_grad():
        res = model(new_noiseSpec, flag='test')
        sf = nn.Softmax(dim=1).to(device)
        res = sf(res)
        res = res.detach().cpu().numpy()[0]
        conf = res[0]
        conf = conf[:ori_h, :ori_w]
        conf[:, :2000] = np.where(conf[:, :2000] < thresh, 0, conf[:, :2000])
        conf[:, 2000:] = np.where(conf[:, 2000:] < 0.4, 0, conf[:, 2000:])
    # print('conf', conf.shape)
    foreground = conf * noiseSpec
    conf[conf >= thresh] = 1
    conf = conf.astype(np.int32)
    
    return foreground, conf
from yolo_denoiser import YoloDenoiser
noise_data_path = 'data/613_spec_data_all'
output_folder = 'data/613_denoise_image_all'
os.makedirs(output_folder, exist_ok=True)
model = YoloDenoiser(out_channel=2)
device = 'cpu'
model_path = f"work_dir/yolo_4feat_deepnoise_log_randomh_613_moremorexianpu_snrr0.05/yolo_deepDenoiser_20_240.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
for name in os.listdir(noise_data_path):
    print('name', name)
    noisy_spec = np.load(noise_data_path + '/' + name)
    noisy_spec = noisy_spec / noisy_spec.std()
    noisy_spec = noisy_spec[:, :10000]
    denoise_noisy_spec, conf = test_deepdenoiser_stft_fragment(model, noisy_spec, thresh=0.1)
    denoise_image = to_image(denoise_noisy_spec)
    print('denoise_image', denoise_image.shape)
    # cv2.imwrite(f"{output_folder}/{name.replace('.npy', 'denoise.png')}", denoise_image)
    # conf_image = to_image(conf
    # cv2.imwrite(f"./{output_folder}/conf_image.png", conf_image)
    # res[res <= 0.5] = 0  # 卡阈值预处理 降低运算量
    trace_img_vis, peaks_img, trajs = extract_multi_lines(denoise_noisy_spec, out_prefix='demo_multi',
                                    prominence=5, min_length=40, delta_f=10, fre_range=10)
    peaks_img = to_image(peaks_img)
    trace_img_vis = to_image(trace_img_vis)
    noisy_spec_image = to_image(noisy_spec)
    ruler = make_freq_ruler(
    width=noisy_spec_image.shape[1],
    height=45,
    px_per_hz=20,
    tick_hz=5
    )
    # 插在三张时频图之间
    all_image = np.vstack([
        ruler,
        noisy_spec_image,
        ruler,
        denoise_image,
        ruler,
        trace_img_vis,
        ruler
    ])
    cv2.imwrite(f"./{output_folder}/{name.replace('.npy', 'spec.png')}", noisy_spec_image[:,0:1000,:])
    cv2.imwrite(f"./{output_folder}/{name.replace('.npy', 'denoise_ori.png')}", denoise_image[:,0:1000,:])
    cv2.imwrite(f"./{output_folder}/{name.replace('.npy', 'peaks_img.png')}", peaks_img[:,0:1000,:])
    cv2.imwrite(f"./{output_folder}/{name.replace('.npy', 'trace_img_vis.png')}", trace_img_vis[:,0:1000,:])
    # all_image = np.vstack([noisy_spec_image, denoise_image, trace_img_vis])
    cv2.imwrite(f"./{output_folder}/{name.replace('.npy', 'denoise.png')}", all_image)
    trace_data = []
    for idx in range(len(trajs)):
            tra = trajs[idx]
            for j in range(len(tra)):
                t, f = tra[j]
                t += 10
                f = f / 20
                info = [idx+1, t ,f]
                trace_data.append(info)
    trace_data = np.array(trace_data)

    # pid = name.split('_Pt')[0]
    # print('pid', pid)
    # data_root = f'G:/jiuzhou/VectorDOA/jiuzhou_613data/{pid}'
    # ptData_file = data_root + f'/{pid}_Pt.wav'
    # vxData_file = data_root + f'/{pid}_Vx.wav'
    # vyData_file = data_root + f'/{pid}_Vy.wav'
    # # lineRecords_file = f'613_denoise_image_all_DOA/{pid}_Pttrace.txt'
    # fs, ptData = wavfile.read(ptData_file)
    # fs, vxData = wavfile.read(vxData_file)
    # fs, vyData = wavfile.read(vyData_file)
    # T = len(ptData) // fs
    # ptData = ptData[: T * fs]
    # vxData = vxData[: T * fs]
    # vyData = vyData[: T * fs]
    # # lineRecords = np.loadtxt(lineRecords_file, delimiter=',')
    # new_lineRecords = vector_doa(ptData, vxData, vyData, fs, trace_data)
    # os.makedirs('613_denoise_image_all_DOA', exist_ok=True)
    # np.savetxt(f'613_denoise_image_all_DOA/{pid}_Pttrace_python_0511.txt', new_lineRecords, delimiter=',', fmt=["%d", "%d", "%.2f", "%.2f"])
    # print('lineRecords', trace_data.shape)
    # print('new_lineRecords', new_lineRecords.shape)

    # break
