import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import socket
import json
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
from doa_python import vector_doa
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

def load_and_concat_wav(file_list, t):
    """Load and concatenate multiple wav files with time constraints

    Args:
        file_list: list of file paths for Pt, Vx, Vy channels (sorted by time)
        t: [start_time, skip_end_time]
            - t[0]: start time in seconds (skip data before this time)
            - t[1]: skip last t[1] seconds from the end of all data

    Returns:
        ptData, vxData, vyData: concatenated audio data
        fs: sample rate
    """
    pt_data_list, vx_data_list, vy_data_list = [], [], []

    # First pass: calculate total duration
    total_duration = 0
    for files in file_list:
        fs, pt = wavfile.read(files[0])
        total_duration += len(pt) / fs

    # Calculate actual end time
    end_time = total_duration - t[1]

    # Second pass: extract data
    cumulative_time = 0
    for i, files in enumerate(file_list):
        fs, pt = wavfile.read(files[0])
        _, vx = wavfile.read(files[1])
        _, vy = wavfile.read(files[2])

        file_duration = len(pt) / fs
        file_start = cumulative_time
        file_end = cumulative_time + file_duration

        # Calculate valid range for this file
        valid_start = max(0, t[0] - file_start)
        valid_end = min(file_duration, end_time - file_start)

        if valid_end > valid_start:
            start_idx = int(valid_start * fs)
            end_idx = int(valid_end * fs)
            pt_data_list.append(pt[start_idx:end_idx])
            vx_data_list.append(vx[start_idx:end_idx])
            vy_data_list.append(vy[start_idx:end_idx])

        cumulative_time = file_end

    ptData = np.concatenate(pt_data_list)
    vxData = np.concatenate(vx_data_list)
    vyData = np.concatenate(vy_data_list)

    return ptData, vxData, vyData, fs

def wav2spec(audio_root_path, pid):
    pt_file = f'{audio_root_path}/{pid}_Pt.wav'
    audio_data, Fs = librosa.load(pt_file, sr=None)
    print(audio_data.shape, Fs)
    T = audio_data.shape[0] // Fs
    print(f'all time : ==================={T}===================')
    print(f'Fs : ==================={Fs}===================')
    audio_data = audio_data[: T * Fs]
    audio_data = audio_data.reshape(-1, Fs)
    spec_data = wave_to_spec_log10(audio_data, frequency_resolution=20, fs=Fs, f_lower_bound=0, f_higher_bound=100)
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
        conf[:, 2000:] = np.where(conf[:, 2000:] < 0.4, 0, conf[:, 2000:])
    # print('conf', conf.shape)
    foreground = conf * noiseSpec
    conf[conf >= thresh] = 1
    conf = conf.astype(np.int32)
    
    return foreground, conf

def wav2image(file_list, t, model_path, out_dir='vis_temp', device='cpu'):
    """Process multiple wav files and generate time-azimuth map

    Args:
        file_list: list of [Pt, Vx, Vy] file paths
        t: time constraints [start_first, ..., end_last]
        model_path: path to denoiser model
        out_dir: output directory for images
        device: computation device

    Returns:
        noisy_spec, denoise_noisy_spec, tf_azimuth_map, azimuth_freq_map
    """
    model = YoloDenoiser(out_channel=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    # Load and concatenate wav files
    ptData, vxData, vyData, fs = load_and_concat_wav(file_list, t)
    T = len(ptData) // fs
    ptData = ptData[: T * fs]
    vxData = vxData[: T * fs]
    vyData = vyData[: T * fs]

    # Convert to spectrogram
    audio_data = ptData.reshape(-1, fs)
    noisy_spec = wave_to_spec_log10(audio_data, frequency_resolution=20, fs=fs, f_lower_bound=0, f_higher_bound=100)
    noisy_spec = noisy_spec[:, :-1]
    noisy_spec = noisy_spec / noisy_spec.std()
    noisy_spec = noisy_spec[:, :2000]

    # Denoise
    denoise_noisy_spec, conf = test_deepdenoiser_stft_fragment(model, noisy_spec, thresh=0.1)

    # Extract lines
    trace_img_vis, peaks_img, trajs = extract_multi_lines(denoise_noisy_spec, out_prefix='demo_multi',
                                    prominence=5, min_length=40, delta_f=10, fre_range=10)
    trace_data = []
    for idx in range(len(trajs)):
        tra = trajs[idx]
        for j in range(len(tra)):
            t_val, f = tra[j]
            t_val += 10
            f = f / 20
            info = [idx+1, t_val, f]
            trace_data.append(info)

    # DOA estimation
    new_lineRecords = vector_doa(ptData, vxData, vyData, fs, trace_data)
    tf_azimuth_map = doa_to_tfmap(new_lineRecords, noisy_spec.shape)
    azimuth_freq_map = tf_to_azimuth_freq(denoise_noisy_spec, tf_azimuth_map)

    # Save images
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f'{out_dir}/spec_image.png', to_image(noisy_spec))
    cv2.imwrite(f'{out_dir}/denoise_image.png', to_image(denoise_noisy_spec))
    cv2.imwrite(f'{out_dir}/tf_azimuth_map_image.png', azimuth_to_image(tf_azimuth_map))
    cv2.imwrite(f'{out_dir}/time_azimuth_image.png', to_image(azimuth_freq_map))

    # Save mat file
    io.savemat(f'{out_dir}/results.mat', {
        'noisy_spec': noisy_spec,
        'denoise_noisy_spec': denoise_noisy_spec,
        'tf_azimuth_map': tf_azimuth_map,
        'azimuth_freq_map': azimuth_freq_map
    })

    return noisy_spec, denoise_noisy_spec, tf_azimuth_map, azimuth_freq_map

if __name__=='__main__':
    HOST = '127.0.0.1'
    PORT = 19999

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print(f'Socket server started on {HOST}:{PORT}')

    while True:
        client_socket, addr = server_socket.accept()
        print(f'Connected by {addr}')

        try:
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b'\n\n' in data:
                    break

            request = json.loads(data.decode('utf-8'))
            head = request.get('head', {})

            if head.get('id') != 'scjz':
                response = {
                    'status': 'ignored',
                    'message': 'Invalid id, request ignored'
                }
            else:
                params = request.get('params', {})

                file_name = params.get('file_name')
                t = params.get('t')
                out_dir = params.get('OutDir', 'vis_temp')
                model_path = params.get('model_path')

                noisy_spec, denoise_noisy_spec, tf_azimuth_map, azimuth_freq_map = wav2image(
                    file_name, t, model_path, out_dir=out_dir, device='cpu'
                )

                response = {
                    'status': 'success',
                    'message': 'Processing completed',
                    'output_dir': out_dir
                }

        except Exception as e:
            response = {
                'status': 'error',
                'message': str(e)
            }

        client_socket.sendall(json.dumps(response).encode('utf-8'))
        client_socket.close()

