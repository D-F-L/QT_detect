import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

def wave_to_spec_log10(wave, frequency_resolution, fs, f_lower_bound, f_higher_bound):
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

def to_image(data, type='RGB'):
    # p_low = np.percentile(data, 5)
    # p_high = np.percentile(data, 95)
    # # 数据到5%-95%范围
    # data_clipped = np.clip(data, p_low, p_high)
    data_clipped = data.copy()
    image = ((data_clipped - data_clipped.min()) * 255 / (data_clipped.max() - data_clipped.min())).astype(np.uint8)
    if type=='RGB':
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image

audio_root_path = 'G:/jiuzhou/VectorDOA/jiuzhou_613data'
audio_files = []
for root, _, files in os.walk(audio_root_path):
    for file in files:
        if 'Pt.wav' in file:
            audio_files.append(root + '/' + file)
    # audio_path = '/data/sdv3/xiajinsong/jiuzhou/data/613/20230719_133454/20230719_133454._39_Pt.wav'
for audio_path in audio_files:
    print('audio_path', audio_path)
    audio_data, Fs = librosa.load(audio_path, sr=None)
    save_png = 'data/613_spec_image_360'
    # save_gray_png = 'data/613_gray_image'
    save_data = 'data/613_spec_data_360'
    os.makedirs(save_png, exist_ok=True)
    # os.makedirs(save_gray_png, exist_ok=True)
    os.makedirs(save_data, exist_ok=True)
    print(audio_data.shape, Fs)
    T = audio_data.shape[0] // Fs
    print(f'all time : ==================={T}===================')
    print(f'Fs : ==================={Fs}===================')
    for i in range(0, T, T):
        print(f'==================={i}===================')
        audio_data_split = audio_data[: T * Fs]
        audio_data_split = audio_data_split[: 360 * Fs]
        # if len(audio_data_split) < 360 * Fs:
        #     continue
        audio_data_split = audio_data_split.reshape(-1, Fs)
        spec_data = wave_to_spec_log10(audio_data_split, frequency_resolution=20, fs=Fs, f_lower_bound=0, f_higher_bound=100)
        spec_data = spec_data[:, :-1]
        print('spec_data', spec_data.shape)
        img = to_image(spec_data)
        gray_img = to_image(spec_data, type='GRAY')
        file_name = audio_path.split('/')[-1]
        np.save(save_data + '/' + file_name.replace('.wav',  f'.npy'), spec_data)
        cv2.imwrite(save_png + '/' + file_name.replace('.wav', f'.png'), img)
        # cv2.imwrite(save_gray_png + '/' + file_name.replace('.wav', f'_{i}.png'), gray_img)