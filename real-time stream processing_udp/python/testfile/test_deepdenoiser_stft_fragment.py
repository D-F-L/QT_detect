import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
from scipy import io
from psignal.lofar import draw, norm, spec_single_stft, spec_single_stft_log10
from models.deepnoise_unet import DeepDenoiser

# -------------------------------
# 主函数：接收参数并执行
# -------------------------------
def test_deepdenoiser_stft_fragment(audio_path, frequency_resolution, epochs, f_low, f_high, start_time, end_time):
    output_folder = "z_output"
    os.makedirs(output_folder, exist_ok=True)

    # 加载音频
    if audio_path.endswith(".wav"):
        audio_data, Fs = librosa.load(audio_path, sr=None)
    elif audio_path.endswith(".txt"):
        audio_data = np.loadtxt(audio_path, delimiter=",")
        Fs =16000
    
    print(len(audio_data))
    audio_data = audio_data.reshape(-1)
    audio_data = audio_data[start_time * Fs:end_time * Fs]
    T = end_time - start_time
    assert audio_data.shape[0] == T * Fs

    noiseWave = audio_data.reshape((T, Fs))
    print(noiseWave.shape)
    noiseWave = norm(noiseWave)
    noiseSpec = spec_single_stft_log10(noiseWave, f_low, f_high, frequency_resolution, Fs)
    noiseSpec = norm(noiseSpec)
    noiseSpec = noiseSpec / np.std(noiseSpec)

    # 初始化模型
    model = DeepDenoiser()
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    foreground_list = []
    noiseWave_list = []
    for epoch in epochs:
        model_path = f"/data/sdv1/xiangrui/denoiser/work_dir/orignal_deepnoise/deepDenoiser_20_1980.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        noiseSpec_t = torch.tensor(noiseSpec).cuda().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            res = model(noiseSpec_t)
            sf = nn.Softmax(dim=1).cuda()
            res = sf(res)
            res = res.detach().cpu().numpy()[0]
        # 置信度后处理res[0]?不好卡 且结果杂乱
        conf = res[0]
        conf[conf < 0.7] = 0
        foreground = conf * noiseSpec
        # background = res[1] * noiseSpec
        epoch_folder = os.path.join(output_folder, f"epoch_{epoch}")
        os.makedirs(epoch_folder, exist_ok=True)
        
        # 保存 .mat 文件
        io.savemat(os.path.join(epoch_folder, f"stft_{frequency_resolution}s_{epoch}epoch.mat"), {
            f'res{epoch}': conf,
            f'before{epoch}': noiseSpec,
            f'after{epoch}': foreground
        })

        # 保存图像
        draw(foreground, f_low, f_high, frequency_resolution,
             os.path.join(epoch_folder, f"after_stft_{frequency_resolution}s_{epoch}epoch.png"))
        draw(noiseSpec, f_low, f_high, frequency_resolution,
             os.path.join(epoch_folder, f"before_stft_{frequency_resolution}s_{epoch}epoch.png"))
        
        # 保存频谱总和图
        plt.figure()
        plt.plot(noiseSpec[:, :].sum(0))
        plt.title('Before Denoising Spectrum Sum')
        plt.savefig(os.path.join(epoch_folder, f"before_stft_sum_{frequency_resolution}s_{epoch}epoch.png"))
        plt.close()

        plt.figure()
        plt.plot(foreground[:, :].sum(0))
        plt.title('After Denoising Spectrum Sum')
        plt.savefig(os.path.join(epoch_folder, f"after_stft_sum_{frequency_resolution}s_{epoch}epoch.png"))
        plt.close()

        print(f"[INFO] Epoch {epoch} completed. Results saved to {epoch_folder}")
        foreground_list.append(foreground)
        noiseWave_list.append(noiseWave)
    return noiseWave_list
