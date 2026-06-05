import torch
import numpy as np

def vector_doa(ptData, vxData, vyData, fs, lineRecords, device='cpu'):
    """
    Torch优化版本的VectorDOA，使用矩阵运算加速

    参数
    ----
    device : str
        计算设备，'cuda' 或 'cpu'，默认 'cuda'
    """
    # 输入转换
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

    # 转换为torch张量
    # device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    pdata = torch.from_numpy(ptData).float().to(device)
    vxdata = torch.from_numpy(vxData).float().to(device)
    vydata = torch.from_numpy(vyData).float().to(device)

    # 归一化
    pdata = pdata / (torch.abs(pdata).max() + 1e-10)
    vxdata = vxdata / (torch.abs(vxdata).max() + 1e-10)
    vydata = vydata / (torch.abs(vydata).max() + 1e-10)

    # 基本参数
    T = len(ptData) / fs
    win_Len = 10
    nbins = 360 * 2
    Nfft = win_Len * fs

    theta = np.linspace(0, 360, nbins + 1)
    bin_centers = (theta[:-1] + theta[1:]) / 2
    theta_torch = torch.from_numpy(theta).float().to(device)
    bin_centers_torch = torch.from_numpy(bin_centers).float().to(device)

    def matlab_round_pos(x):
        return np.floor(np.asarray(x) + 0.5).astype(int)

    # 按目标ID分组
    ids = lineRecords[:, 0]
    unique_ids = []
    groups = {}

    for idx, id_val in enumerate(ids):
        if id_val not in groups:
            groups[id_val] = []
            unique_ids.append(id_val)
        groups[id_val].append(idx)

    # 针对每个目标ID进行DOA估计
    for id_val in unique_ids:
        row_indices = np.array(groups[id_val], dtype=int)
        tar_tf = lineRecords[row_indices, 1:3]
        M = tar_tf.shape[0]

        # 时间窗口
        tar_time_center = tar_tf[:, 0]
        tar_time = np.column_stack([
            np.maximum(tar_time_center - 5, 1),
            np.minimum(tar_time_center + 5, T)
        ])

        # 频率带
        tar_freq_center = tar_tf[:, 1]
        tar_freq = np.column_stack([
            np.maximum(tar_freq_center - 0.5, 10),
            tar_freq_center + 0.5
        ])

        fLn_array = np.maximum(1, matlab_round_pos(tar_freq[:, 0] * Nfft / fs))
        fHn_array = np.minimum(Nfft // 2, matlab_round_pos(tar_freq[:, 1] * Nfft / fs))

        for n in range(M):
            t_range = tar_time[n, :]
            start_idx = max(int(np.floor(t_range[0] * fs)), 0)
            end_idx = min(int(np.floor(t_range[1] * fs)), len(ptData))

            if end_idx <= start_idx or end_idx - start_idx < win_Len * fs:
                continue

            # 提取信号段
            p = pdata[start_idx:end_idx]
            x = vxdata[start_idx:end_idx]
            y = vydata[start_idx:end_idx]

            sig_len = len(p)
            Nfft_local = min(Nfft, sig_len)

            # Hanning窗（使用numpy保持一致性）
            win = torch.from_numpy(np.hanning(sig_len)).float().to(device)

            # FFT（批量处理3个信号）
            signals = torch.stack([p * win, x * win, y * win])
            ffts = torch.fft.fft(signals, n=Nfft_local, dim=1) / sig_len
            p_fft, x_fft, y_fft = ffts[0], ffts[1], ffts[2]

            half_nfft = Nfft_local // 2
            fLn = min(fLn_array[n], half_nfft)
            fHn = min(fHn_array[n], half_nfft)

            if fLn > fHn:
                continue

            # 提取频段
            p_seg = p_fft[fLn - 1:fHn]
            x_seg = x_fft[fLn - 1:fHn]
            y_seg = y_fft[fLn - 1:fHn]

            if len(p_seg) == 0:
                continue

            # 交叉谱
            Pvx2 = torch.real(p_seg * torch.conj(x_seg))
            Pvy2 = torch.real(p_seg * torch.conj(y_seg))

            # 角度估计
            est_angle = torch.atan2(Pvy2, Pvx2) * 180 / np.pi

            # unwrap
            est_angle_np = est_angle.cpu().numpy()
            theta_unwrap_rad = np.unwrap(np.radians(est_angle_np))
            est_angle_np = np.degrees(theta_unwrap_rad)
            est_angle_np = np.mod(est_angle_np + 180, 360)
            est_angle = torch.from_numpy(est_angle_np).float().to(device)

            # 按幅度加权
            Af = torch.abs(p_seg)

            if torch.all(Af == 0):
                theta_weight = -5
            else:
                max_Af = torch.max(Af)

                if max_Af == 0 or not torch.isfinite(max_Af):
                    theta_weight = -5
                else:
                    AddN = Af / max_Af

                    # 使用torch.searchsorted进行bin分配
                    bin_idx = torch.searchsorted(theta_torch, est_angle, right=False)
                    bin_idx = torch.clamp(bin_idx, 0, nbins - 1)

                    valid = (
                        torch.isfinite(est_angle) &
                        torch.isfinite(AddN) &
                        (bin_idx >= 0) &
                        (bin_idx < nbins)
                    )

                    if torch.any(valid):
                        count_weight = torch.zeros(nbins, dtype=torch.float32, device=device)

                        # 使用scatter_add进行累加
                        count_weight.scatter_add_(0, bin_idx[valid], AddN[valid])

                        max_idx_weight = torch.argmax(count_weight)
                        theta_weight = bin_centers_torch[max_idx_weight].item()
                    else:
                        theta_weight = -5

            lineRecords[row_indices[n], 3] = theta_weight

    return lineRecords
