import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
from scipy import io
from psignal.lofar import draw, norm, spec_single_stft, spec_single_stft_log10
from models.deepnoise_unet import DeepDenoiser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"
# -------------------------------
# 主函数：接收参数并执行
# -------------------------------
def run_deepdenoiser_slide(
    audio_path,
    frequency_resolution,
    epochs,
    f_low,
    f_high,
    T,
    hop_length,
    mode='normal'
):
    output_folder = "z_output"
    os.makedirs(output_folder, exist_ok=True)

    # 1. 载入音频并切块
    if audio_path.endswith(".wav"):
        audio_data, Fs = librosa.load(audio_path, sr=None)
    elif audio_path.endswith(".txt"):
        audio_data = np.loadtxt(audio_path, delimiter=",")
        Fs = 16000
    time_sec = int(np.floor(len(audio_data) / (Fs * T)))
    audio_data = audio_data[:time_sec * Fs * T]

    # 2. 初始化模型
    model = DeepDenoiser()
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    result = []

    for epoch in epochs:
        if mode != 'normal':
            model.load_state_dict(
            torch.load(
                f"/data/sdv1/xiangrui/denoiser/work_dir/orignal_deepnoise/deepDenoiser_20_1440.pth",  # 实地测试需要换一下
                map_location=torch.device('cpu')
                )
            )
            combined_data = []
            # 3. 滑窗处理
            n_slices = int((len(audio_data) / Fs - T) / hop_length) + 1
            for i in range(n_slices):
                print(i)
                if i == 0:
                    noiseWave = audio_data[:T * Fs]
                else:
                    noiseWave = audio_data[i * hop_length * Fs:(i * hop_length + T) * Fs]
                noiseWave = norm(noiseWave)

                freq_list = []

                n_freq_bins = int((f_high - f_low) / (frequency_resolution * 50))
                for j in range(n_freq_bins):
                    start_f = j * frequency_resolution * 50
                    end_f = (j + 1) * frequency_resolution * 50

                    noiseSpec = spec_single_stft(noiseWave, start_f, end_f, frequency_resolution, Fs)
                    noiseSpec = norm(noiseSpec)
                    noiseSpec = noiseSpec / np.std(noiseSpec)

                    # 模型推理
                    noiseSpec_t = torch.tensor(noiseSpec).cuda().unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        res = model(noiseSpec_t)
                        sf = nn.Softmax(dim=1).cuda()
                        res = sf(res)
                        res = res.detach().cpu().numpy()[0]

                    foreground = res[0] * noiseSpec

                    if j == 0:
                        freq_list = foreground
                    else:
                        freq_list = np.hstack((freq_list, foreground))
                combined_data.append(freq_list)

            # 4. 拼接最终结果（每秒取最大值）
            total_seconds = int(len(audio_data) / Fs)
            after_list = np.zeros((total_seconds, combined_data[0].shape[1]))

            for i in range(total_seconds):
                win_data_list = []
                if i <= 40:
                    for j in range(0, i + 1):
                        win_data = combined_data[j][i - j:i - j + 1, :]
                        win_data_list.append(win_data[0])
                elif 40 < i <= 80:
                    for j in range(i - 40, i + 1):
                        win_data = combined_data[j][i - j:i - j + 1, :]
                        win_data_list.append(win_data[0])
                else:
                    for j in range(i, total_seconds):
                        win_data = combined_data[j - 40][40 - (j - i):40 - (j - i) + 1, :]
                        win_data_list.append(win_data[0])
                if mode =='max':
                    col_maxs = [max(col) for col in zip(*win_data_list)]
                    after_list[i:i + 1, :] = col_maxs
                elif mode =='mean':
                    col_means = [sum(col) / len(col) for col in zip(*win_data_list)]
                    after_list[i:i + 1, :] = col_means

            # 5. 保存结果
            result.append(after_list)


        # 普通情况 不进行多滑动窗口叠加求平均或最大值 滑动完直接拼接
        else:
            model.load_state_dict(
            torch.load(
                f"/data/sdv1/xiangrui/denoiser/work_dir/orignal_deepnoise/deepDenoiser_20_1440.pth",
                map_location=torch.device('cpu')
                )
            )
            combined_data = []
            # 3. 滑窗处理
            n_slices = int((len(audio_data) / Fs - T) / hop_length) + 1
            for i in range(n_slices):
                print(i)
                if i == 0:
                    noiseWave = audio_data[:T * Fs]
                else:
                    noiseWave = audio_data[i * hop_length * Fs:(i * hop_length + T) * Fs]
                noiseWave = norm(noiseWave)
                freq_list = []

                n_freq_bins = int((f_high - f_low) / (frequency_resolution * 50))
                for j in range(n_freq_bins):
                    start_f = j * frequency_resolution * 50
                    end_f = (j + 1) * frequency_resolution * 50

                    noiseSpec = spec_single_stft(noiseWave, start_f, end_f, frequency_resolution, Fs)
                    noiseSpec = norm(noiseSpec)
                    noiseSpec = noiseSpec / np.std(noiseSpec)

                    # 推理
                    noiseSpec_t = torch.tensor(noiseSpec).cuda().unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        res = model(noiseSpec_t)
                        sf = nn.Softmax(dim=1).cuda()
                        res = sf(res)
                        res = res.detach().cpu().numpy()[0]

                    foreground = res[0] * noiseSpec
                    # foreground = 20*np.log10(abs(foreground))

                    if j == 0:
                        freq_list = foreground
                    else:
                        freq_list = np.hstack((freq_list, foreground))

                if i == 0:
                    combined_data = freq_list
                else:
                    combined_data = np.vstack((combined_data, freq_list[-hop_length:, :]))
            result.append(combined_data)

    output_folder = "z_output"
    os.makedirs(output_folder, exist_ok=True)
    epoch_folder = os.path.join(output_folder, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)
        
    # 保存 .mat 文件
    io.savemat(os.path.join(epoch_folder, f"stft_{frequency_resolution}s_{epoch}epoch.mat"), {
        f'result{epoch}': result[0],
    })
    return result