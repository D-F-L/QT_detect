import numpy as np
from scipy.io import wavfile

import numpy as np


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

pid = '20221126_104650'
data_root = f'D:/data/jiuzhou_613data/{pid}'
ptData_file = data_root + '/Pt.wav'
vxData_file = data_root + '/Vx.wav'
vyData_file = data_root + '/Vy.wav'
lineRecords_file = f'613_denoise_image_all_DOA/{pid}_Pttrace.txt'
fs, ptData = wavfile.read(ptData_file)
fs, vxData = wavfile.read(vxData_file)
fs, vyData = wavfile.read(vyData_file)
T = len(ptData) // fs
ptData = ptData[: T * fs]
vxData = vxData[: T * fs]
vyData = vyData[: T * fs]
lineRecords = np.loadtxt(lineRecords_file, delimiter=',')
new_lineRecords = vector_doa(ptData, vxData, vyData, fs, lineRecords)
np.savetxt(f'613_denoise_image_all_DOA/{pid}_Pttrace_python.txt', new_lineRecords, delimiter=',', fmt=["%d", "%d", "%.2f", "%.2f"])
print('lineRecords', lineRecords.shape)
print('new_lineRecords', new_lineRecords.shape)

