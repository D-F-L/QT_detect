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
import cv2
from testfile.extract_multi_lines import extract_multi_lines
from yolo_denoiser import YoloDenoiser
from scipy.io import wavfile
from doa_python_optimized import vector_doa
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
    if len(data.shape) == 2:
        if type == 'RGB':
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image

def add_azimuth_scale_bar(image, tick_step=10, bar_height=30):
    """Add an azimuth scale bar on top of the image.

    Args:
        image: input image with width representing azimuth 0-360 degrees
        tick_step: tick interval in degrees
        bar_height: height of the scale bar in pixels

    Returns:
        new image with scale bar on top
    """
    h, w = image.shape[:2]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bar = np.full((bar_height, w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    for deg in range(0, 361, tick_step):
        x = int(round(deg * (w - 1) / 360.0))
        x = min(max(x, 0), w - 1)
        # Major ticks every 60 degrees, minor otherwise
        is_major = (deg % 60 == 0)
        tick_len = 10 if is_major else 5
        cv2.line(bar, (x, bar_height - 1), (x, bar_height - 1 - tick_len), (0, 0, 0), 1)
        if is_major:
            label = str(deg)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx = x - tw // 2
            tx = min(max(tx, 0), w - tw)
            ty = th + 2
            cv2.putText(bar, label, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Border line between bar and image
    cv2.line(bar, (0, bar_height - 1), (w - 1, bar_height - 1), (0, 0, 0), 1)
    return np.vstack([bar, image])

def azimuth_to_image(azimuth_map):
    """Visualize azimuth map with HSV colormap

    Args:
        azimuth_map: array with azimuth values (0-360) or -5 for no data

    Returns:
        RGB image where hue represents azimuth angle
    """
    h, w = azimuth_map.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # Mask valid data (not -5)
    valid_mask = azimuth_map >= 0

    # Hue: 0-360 degrees -> 0-179 (OpenCV HSV range)
    hsv[:, :, 0] = np.where(valid_mask, (azimuth_map / 2).astype(np.uint8), 0)
    # Saturation: full for valid data, 0 for invalid
    hsv[:, :, 1] = np.where(valid_mask, 255, 0)
    # Value: full for valid data, dark for invalid
    hsv[:, :, 2] = np.where(valid_mask, 255, 30)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def doa_to_tfmap(doa_records, spec_shape):
    """Convert DOA records to time-frequency-azimuth map

    Args:
        doa_records: array with shape (N, 4), columns: [id, time, freq, azimuth]
        spec_shape: tuple (time_bins, freq_bins), shape of spectrogram

    Returns:
        tf_azimuth_map: array with shape (time_bins, freq_bins), values are azimuth angles
    """
    time_bins, freq_bins = spec_shape
    tf_map = np.full((time_bins, freq_bins), -5.0)

    for record in doa_records:
        _, t, f, azimuth = record
        t_idx = int(t) - 10
        f_idx = int(f * 20)

        if 0 <= t_idx < time_bins and 0 <= f_idx < freq_bins:
            tf_map[t_idx, f_idx] = azimuth

    return tf_map

def tf_to_azimuth_freq(noisy_spec, tf_azimuth_map):
    """Convert time-frequency map to time-azimuth map

    Args:
        noisy_spec: spectrogram with shape (time_bins, freq_bins)
        tf_azimuth_map: azimuth map with shape (time_bins, freq_bins)

    Returns:
        time_azimuth_map: array with shape (time_bins, 360), values are intensity
    """
    time_bins, freq_bins = noisy_spec.shape
    time_azimuth_map = np.zeros((time_bins, 360), dtype=np.float32)

    for t in range(time_bins):
        for f in range(freq_bins):
            azimuth = tf_azimuth_map[t, f]
            if azimuth >= 0:
                azimuth_idx = int(azimuth) % 360
                time_azimuth_map[t, azimuth_idx] += noisy_spec[t, f]

    return time_azimuth_map

def wav2spec(audio_root_path, pid):
    pt_file = f'{audio_root_path}/{pid}_Pt.wav'
    audio_data, Fs = librosa.load(pt_file, sr=None)
    print(audio_data.shape, Fs)
    T = audio_data.shape[0] // Fs
    print(f'all time : ==================={T}===================')
    print(f'Fs : ==================={Fs}===================')
    audio_data = audio_data[: T * Fs]
    audio_data = audio_data.reshape(-1, Fs)
    spec_data = wave_to_spec_log10(audio_data, frequency_resolution=20, fs=Fs, f_lower_bound=0, f_higher_bound=500)
    spec_data = spec_data[:, :-1]
    print('spec_data', spec_data.shape)
    return spec_data

def test_deepdenoiser_stft_fragment(model, noiseSpec, thresh=0.01, device='cpu'):
    ori_h, ori_w = noiseSpec.shape
    target_h, target_w = (ori_h - 1) // 32 + 1, (ori_w - 1) // 32 + 1
    new_noiseSpec = np.zeros((target_h * 32, target_w * 32), dtype=np.float32)
    new_noiseSpec[:ori_h, :ori_w] = noiseSpec.astype(np.float32)
    new_noiseSpec = torch.tensor(new_noiseSpec).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)  # → [3, H, W]
    new_noiseSpec = new_noiseSpec.float()
    with torch.no_grad():
        res = model(new_noiseSpec, flag='test')
        sf = nn.Softmax(dim=1).to(device)
        res = sf(res)
        res = res.detach().cpu().numpy()[0]
        conf = res[0]
        conf = conf[:ori_h, :ori_w]
        conf[:, :2000] = np.where(conf[:, :2000] < thresh, 0, conf[:, :2000])
        conf[:, 2000:] = np.where(conf[:, 2000:] < 0.1, 0, conf[:, 2000:])
    # print('conf', conf.shape)
    foreground = conf * noiseSpec
    conf[conf >= thresh] = 1
    conf = conf.astype(np.int32)
    
    return foreground, conf

def wav2image(audio_root_path, prefix, model_path, device='cpu'):
    model = YoloDenoiser(out_channel=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    noisy_spec = wav2spec(audio_root_path, pid)
    noisy_spec = noisy_spec / noisy_spec.std()
    noisy_spec = noisy_spec[:, :10000]
    denoise_noisy_spec, conf = test_deepdenoiser_stft_fragment(model, noisy_spec, thresh=0.1, device=device)
    # denoise_image = to_image(denoise_noisy_spec)
    # print('denoise_image', denoise_image.shape)

    trace_img_vis, peaks_img, trajs = extract_multi_lines(denoise_noisy_spec, out_prefix='demo_multi',
                                    prominence=5, min_length=40, delta_f=10, fre_range=10)
    trace_data = []
    for idx in range(len(trajs)):
        tra = trajs[idx]
        for j in range(len(tra)):
            t, f = tra[j]
            t += 10
            f = f / 20
            info = [idx+1, t ,f]
            trace_data.append(info)

    ptData_file = audio_root_path + f'/{pid}_Pt.wav'
    vxData_file = audio_root_path + f'/{pid}_Vx.wav'
    vyData_file = audio_root_path + f'/{pid}_Vy.wav'
    # lineRecords_file = f'613_denoise_image_all_DOA/{pid}_Pttrace.txt'
    fs, ptData = wavfile.read(ptData_file)
    fs, vxData = wavfile.read(vxData_file)
    fs, vyData = wavfile.read(vyData_file)
    T = len(ptData) // fs
    ptData = ptData[: T * fs]
    vxData = vxData[: T * fs]
    vyData = vyData[: T * fs]
    new_lineRecords = vector_doa(ptData, vxData, vyData, fs, np.array(trace_data), device=device)
    # Convert DOA results to time-frequency-azimuth map
    tf_azimuth_map = doa_to_tfmap(new_lineRecords, noisy_spec.shape)
    azimuth_freq_map = tf_to_azimuth_freq(denoise_noisy_spec, tf_azimuth_map)

    print('noisy_spec', noisy_spec.shape)
    print('denoise_noisy_spec', denoise_noisy_spec.shape)
    print('trace_img_vis', trace_img_vis.shape)
    print('new_lineRecords', new_lineRecords.shape)
    print('tf_azimuth_map', tf_azimuth_map.shape)
    print('azimuth_freq_map', azimuth_freq_map.shape)

    save_root_path = 'vis_temp_torch'
    os.makedirs(save_root_path, exist_ok=True)
    spec_image = to_image(noisy_spec)
    denoise_image = to_image(denoise_noisy_spec)
    trace_img_vis = to_image(trace_img_vis)
    tf_azimuth_map_image = azimuth_to_image(tf_azimuth_map)
    azimuth_time_image = to_image(azimuth_freq_map)
    azimuth_time_image = add_azimuth_scale_bar(azimuth_time_image, tick_step=10, bar_height=30)
    cv2.imwrite(f'{save_root_path}/{pid}_spec_image.png', spec_image)
    cv2.imwrite(f'{save_root_path}/{pid}_denoise_image.png', denoise_image)
    cv2.imwrite(f'{save_root_path}/{pid}_trace_img_vis.png', trace_img_vis)
    cv2.imwrite(f'{save_root_path}/{pid}_tf_azimuth_map_image.png', tf_azimuth_map_image)
    cv2.imwrite(f'{save_root_path}/{pid}_azimuth_time_image.png', azimuth_time_image)
    np.savetxt(f'{save_root_path}/{pid}_lineRecords.txt', new_lineRecords, delimiter=',', fmt=["%d", "%d", "%.2f", "%.2f"])
    return noisy_spec, denoise_noisy_spec, tf_azimuth_map, azimuth_freq_map
   
from time import time

if __name__=='__main__':
    s = time()
    audio_root_path = 'E:/DLF/DeskTop/jiuzhou_spec/jiuzhou_pro/jiuzhou_613data/20230622_114508'
    pid = '20230622_114508'
    model_path = f"work_dir/yolo_4feat_deepnoise_log_randomh_613_moremorexianpu_snrr0.05/yolo_deepDenoiser_20_240.pth"
    device = 'cpu'
    noisy_spec, denoise_noisy_spec, tf_azimuth_map, azimuth_freq_map = wav2image(audio_root_path, pid, model_path, device=device)
    print('花费时间：', time() - s)