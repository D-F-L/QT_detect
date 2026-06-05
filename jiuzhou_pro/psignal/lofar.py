from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

#orignal lofar code make a copy here
def lofarv1(beamDataOfTarget, hamming_win, SIGNAL_SAMPLERATE):
    fs = SIGNAL_SAMPLERATE
    signallength = SIGNAL_SAMPLERATE
    framelength = SIGNAL_SAMPLERATE
    framenoverlap = 0
    n = framelength
    Not = (signallength-framenoverlap) // (framelength-framenoverlap)
    if Not-(signallength-framenoverlap)/(framelength-framenoverlap) < 0.05 and Not-(signallength-framenoverlap)/(framelength-framenoverlap) > 0.000000001:
        Not += 1

    PowerSD = [0.0]*int(n/2+1)
    h = np.empty(n, dtype=np.complex128)
    midput = np.empty(n, dtype=np.float64)
    for j in range(Not):
        for i in range(n):
            midput[i] = beamDataOfTarget[i+j *(framelength-framenoverlap)]*hamming_win[i]
            h[i] = midput[i] + 1.j *0
        # end for
        h = np.fft.fft(h, framelength)

        PowerSDframe = np.abs(h)
        # PowerSDframe =  h.real[:]**2 + h.imag[:]**2

        for i in range(int(n/2+1)):
            PowerSD[i] = PowerSD[i] + PowerSDframe[i]

    Kmu = 0.0
    for i in range(framelength):
        Kmu = hamming_win[i]*hamming_win[i]+Kmu
    Kmu = Kmu*float(Not)

    Freq = np.array(list(range(int(n/2+1))))/framelength*fs
    PowerSD = PowerSD/Kmu

    PowerSD[0] = PowerSD[0]
    if n % 2 == 0:
        PowerSD[int(np.floor(float(n)/2.0))] = PowerSD[int(np.floor(float(n)/2.0))]/2

    # 去前面的直流分量、进行归一化计算
    signal_lofar_fl = 10.0          # 在此频率之前的频率置为Hz时的值
    signal_fl_th = int(np.floor(
        signal_lofar_fl / (SIGNAL_SAMPLERATE / framelength)))
    PowerSD[:signal_fl_th] = PowerSD[signal_fl_th]

    if np.max(PowerSD[:int(fs/2+1)]) != 0:
        PowerSD[:int(fs/2+1)] = PowerSD[:int(fs/2+1)] / np.max(PowerSD[:int(fs/2+1)])

    return PowerSD


def lofarv2(wave_frame, win, min_value=10.0): 
    """Pnly use 1s wave frame and hamming window length equals to wave len

    Args:
        wave_frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    frame_len = len(wave_frame)
    h = wave_frame * win
    h = np.fft.fft(h, frame_len)
    PowerSDframe = np.abs(h)
    PowerSDframe = PowerSDframe[:int(frame_len/2+1)]
    Kmu = np.sum(win)
    PowerSD = PowerSDframe/Kmu
    PowerSD[int(np.floor(float(frame_len)/2.0))] = PowerSD[int(np.floor(float(frame_len)/2.0))]/2
    signal_lofar_fl = min_value          # 在此频率之前的频率置为Hz时的值
    signal_fl_th = int(signal_lofar_fl)
    PowerSD[:signal_fl_th] = PowerSD[signal_fl_th]
    if np.max(PowerSD) != 0:
        PowerSD = PowerSD / np.max(PowerSD)
    return PowerSD


def lofarv3(wave_frame, win): 
    """Pnly use 1s wave frame and hamming window length equals to wave len

    Args:
        wave_frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    frame_len = len(wave_frame)
    h = wave_frame * win
    PowerSDframe = np.fft.fft(h, frame_len)
    # PowerSDframe = np.abs(h)
    # PowerSDframe = PowerSDframe[:int(frame_len/2+1)]
    Kmu = np.sum(win)
    PowerSD = PowerSDframe/Kmu
    return PowerSD


def lofarv3_back(PowerSD, win, fs=5000):
    """Inverse Spectrum back

    Args:
        PowerSD (_type_): _description_
        win (_type_): _description_
        fs (int, optional): _description_. Defaults to 5000.

    Returns:
        _type_: _description_
    """
    Kmu = np.sum(win)
    PowerSDframe = PowerSD * Kmu
    inverse_frame = np.fft.ifft(PowerSDframe, fs)
    inverse_frame = inverse_frame/win
    return inverse_frame

def manual_spectrogram(x, window, nperseg, noverlap, nfft=None):
    """
        手动实现 Spectrogram
        参数:
            x (array): 输入信号
            fs (int): 采样率
            window (array): (窗函数长度需等于 nperseg)
            nperseg (int): 每帧长度（窗口长度）
            noverlap (int): 重叠样本数
            nfft (int): FFT 点数（默认等于 nperseg)
        返回:
            f (array): 频率轴(Hz)
            t (array): 时间轴（秒）
            Sxx (array): 复数频谱矩阵（形状: [频率点数, 时间帧数])
            SxP (array): 功率谱密度
        """

    # 参数校验
    if len(window) != nperseg:
        raise ValueError("窗口长度必须等于 nperseg")
    if noverlap >= nperseg:
        raise ValueError("noverlap 必须小于 nperseg")

    # 初始化参数
    nfft = nperseg if nfft is None else nfft
    step = nperseg - noverlap
    n_frames = 1 + (len(x) - nperseg) // step

    # 分帧（与 MATLAB 完全一致的索引计算）
    indices = np.arange(nperseg)[None, :] + step * np.arange(n_frames)[:, None]
    frames = x[indices]

    # 加窗（确保使用对称窗口）
    frames_windowed = frames * window.reshape(1, -1)
    min_value = 10
    # FFT 计算（单边频谱）
    Sxx = np.fft.fft(frames_windowed, n=nfft, axis=1)[:, :nfft // 2 + 1]
    stft = 10 * np.log10(np.abs(Sxx) ** 2) * 2
    frame_len = len(stft[0])
    wave = np.zeros((len(stft),frame_len), dtype=np.float64)
    Kmu = np.sum(window)
    for i in range(0,len(stft)):
        PowerSD = stft[i]/Kmu
        PowerSD[int(np.floor(float(frame_len)/2.0))] = PowerSD[int(np.floor(float(frame_len)/2.0))]/2
        signal_lofar_fl = min_value          # 在此频率之前的频率置为Hz时的值
        signal_fl_th = int(signal_lofar_fl)
        PowerSD[:signal_fl_th] = PowerSD[signal_fl_th]
        if np.max(PowerSD) != 0:
            PowerSD = PowerSD / np.max(PowerSD)
        wave[i, :] = PowerSD
        
    return wave

def lofar_stft_nseconds(data, fs, time_duration, overlaps, device='cpu', if_center = False):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data).to(device).to(torch.float64)

    if (0 <= overlaps <= 1):
        hop_length = time_duration * fs * (1 - overlaps)
    elif overlaps > 1:
        hop_length = time_duration * fs - overlaps
        if hop_length < 0:
            raise RuntimeError("overlaps cannot bigger than time_duration*fs")
    else:
        raise RuntimeError("overlaps is int or float")
    win_len = int(fs*time_duration)  # 窗口大小 多少采样点
    return torch.stft(data,
                      n_fft = win_len,
                      hop_length = int(hop_length),
                      win_length = win_len,
                      window=torch.hamming_window(win_len).to(device),
                      center = if_center,  # True要padding False不进行padding
                      return_complex=True, ).T

def draw(spec ,f_low, f_high,frequency_resolution, filename=None):
    f_low_index = f_low * frequency_resolution  # 对应频带索引
    f_high_index = f_high * frequency_resolution

    x_ticks = list(range(f_low_index, f_high_index + 1, 50 * frequency_resolution))
    plt.figure()
    plt.pcolormesh(spec,cmap=LinearSegmentedColormap.from_list('parula', np.loadtxt("./data/cm_data.txt", delimiter=",")))
    plt.colorbar()
    plt.xlabel('Frequency')
    plt.ylabel('Time(s)')
    plt.xticks(x_ticks, rotation=60)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)

def norm(t):
    # if np.max(t) == 0:
    #     return t  # 如果最大值为零，直接返回原数组
    t = t / np.max(t)
    return t

def spec_single_stft_log10(wave, f_low, f_high, frequency_resolution, Fs, overlaps = 0.5): 
    assert frequency_resolution >= 1
    time_duration = frequency_resolution  # 每个窗口的时间长度（秒）
    overlaps_ratio = 1 - 1 / frequency_resolution   # 间隔一秒做stf
    overlaps = time_duration * Fs * overlaps_ratio   # 每个窗口重叠数量
    # if frequency_resolution == 20:
    #     overlaps = time_duration * Fs * 0.95  # 窗口重叠比例 间隔一秒做stft
    # elif frequency_resolution == 10:
    #     overlaps = time_duration * Fs * 0.9  # 窗口重叠比例 间隔一秒做stft
    # elif frequency_resolution==1:  # 时间分辨率为一秒的情况 无重叠
    #     overlaps = 0
    device = 'cpu'  # 计算设备
    wave = wave.reshape(-1)
    stft_result = lofar_stft_nseconds(wave, Fs, time_duration, overlaps, device)
    # 可能导致负数产生 考虑不用这个训练一版？
    stft_magnitude = 20 * torch.log10(torch.abs(stft_result))  # 信噪比较低的时候 需要进行log10
    # stft_magnitude = torch.abs(stft_result)  # 需要判断是否进行log10
    n_fft = int(Fs * time_duration)
    freqs = torch.linspace(0, Fs / 2, n_fft // 2 + 1)
    freq_indices = (freqs >= f_low) & (freqs <= f_high)

    return stft_magnitude.numpy()[:, freq_indices]

def spec_single_stft(wave, f_low, f_high, frequency_resolution, Fs, overlaps = 0.5): 
    assert frequency_resolution >= 1
    time_duration = frequency_resolution  # 每个窗口的时间长度（秒）
    overlaps_ratio = 1 - 1 / frequency_resolution   # 间隔一秒做stf
    overlaps = time_duration * Fs * overlaps_ratio   # 每个窗口重叠数量
    # if frequency_resolution == 20:
    #     overlaps = time_duration * Fs * 0.95  # 窗口重叠比例 间隔一秒做stft
    # elif frequency_resolution == 10:
    #     overlaps = time_duration * Fs * 0.9  # 窗口重叠比例 间隔一秒做stft
    # elif frequency_resolution==1:  # 时间分辨率为一秒的情况 无重叠
    #     overlaps = 0
    device = 'cpu'  # 计算设备
    wave = wave.reshape(-1)
    stft_result = lofar_stft_nseconds(wave, Fs, time_duration, overlaps, device)
    # 可能导致负数产生 考虑不用这个训练一版？
    
    stft_magnitude = torch.abs(stft_result)  # 需要判断是否进行log10
    n_fft = int(Fs * time_duration)
    freqs = torch.linspace(0, Fs / 2, n_fft // 2 + 1)
    freq_indices = (freqs >= f_low) & (freqs <= f_high)

    return stft_magnitude.numpy()[:, freq_indices]

def enhance_columns(arr, cols, radius=3, alpha=1.5, eps=1e-4, in_place=False):
    if not isinstance(cols, (list, tuple, np.ndarray)):
        cols = [cols]
    arr_out = arr if in_place else arr.copy()
    H, W = arr_out.shape
    # 1. 生成选中列掩码
    mask = np.zeros(W, dtype=bool)
    for c in cols:
        left  = max(0, c - radius)
        right = min(W, c + radius + 1)
        mask[left:right] = True
    # 2. 全局先乘 eps
    arr_out *= eps
    # 3. 只在 alpha != 1 时覆盖兴趣区
    if alpha != 1 and mask.any():
        patch = arr[:, mask]          # 取原值
        patch_min = patch.min()
        patch_max = patch.max()
        if patch_max > patch_min:
            patch_norm = (patch - patch_min) / (patch_max - patch_min)
            arr_out[:, mask] = patch_norm * (arr.max() - arr.min()) * alpha + arr.min()
        else:
            arr_out[:, mask] = patch * alpha
    elif alpha ==1:
        patch = arr[:, mask]
        arr_out[:, mask] = patch * alpha
    return arr_out