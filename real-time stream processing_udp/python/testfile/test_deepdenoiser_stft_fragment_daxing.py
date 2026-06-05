import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
from scipy import io
from psignal.lofar import draw, norm, spec_single_stft
from models.deepnoise_unet import DeepDenoiser

# -------------------------------
# 主函数：接收参数并执行
# -------------------------------
def test_deepdenoiser_stft_fragment(audio_path, frequency_resolution, epochs, f_low, f_high, start_time, end_time, fs):
    output_folder = "z_output"
    os.makedirs(output_folder, exist_ok=True)

    # 加载音频
    audio_data, Fs = librosa.load(audio_path, sr=fs)
    print(len(audio_data))
    audio_data = audio_data[start_time * Fs:end_time * Fs]
    T = end_time - start_time
    assert audio_data.shape[0] == T * Fs

    noiseWave = audio_data.reshape((T, Fs))
    print(noiseWave.shape)
    noiseWave = norm(noiseWave)
    noiseSpec = spec_single_stft(noiseWave, f_low, f_high, frequency_resolution, Fs)
    noiseSpec = noiseSpec / np.std(noiseSpec)

    # 初始化模型
    model = DeepDenoiser()
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    for epoch in epochs:
        model_path = f"./work_dir/orignal_deepnoise/deepDenoiser_10_{epoch}.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        noiseSpec_t = torch.tensor(noiseSpec).cuda().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            res = model(noiseSpec_t)
            sf = nn.Softmax(dim=1).cuda()
            res = sf(res)
            res = res.detach().cpu().numpy()[0]

        foreground = res[0] * noiseSpec
        # background = res[1] * noiseSpec
        return foreground