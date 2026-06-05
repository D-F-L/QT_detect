import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
from scipy import io
from psignal.lofar import draw, norm, spec_single_stft, spec_single_stft_log10
from models.deepnoise_unet import DeepDenoiser
from yolo_denoiser import YoloDenoiser
import cv2
from testfile.extract_multi_lines import extract_multi_lines
from yolo_denoiser_3feat import YoloDenoiser3feat
# -------------------------------
# 主函数：接收参数并执行
# -------------------------------

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

def test_deepdenoiser_stft_fragment(noiseSpec, mask, model_name='Y', thresh=0.3):
    flag = 'train'
    output_folder = "z_output_fakedata_zhuanli/" + flag + "/images"
    if model_name == 'Unet':
        # 初始化模型
        model = DeepDenoiser()
        model = nn.DataParallel(model)
        model.cuda()
        model.eval()
        foreground_list = []
        noiseWave_list = []
        model_path = f"/data/sdv1/xiangrui/denoiser/work_dir/orignal_deepnoise/deepDenoiser_20_1980.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        noiseSpec_t = torch.tensor(noiseSpec).cuda().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            res = model(noiseSpec_t)
            sf = nn.Softmax(dim=1).cuda()
            res = sf(res)
            res = res.detach().cpu().numpy()[0]
            conf = res[0]
            entroy_loss = - (mask * np.log(conf) + (1 - mask) * np.log(1 -conf))
            entroy_loss = np.mean(entroy_loss)
            conf[conf < thresh] = 0
    else:
        # 初始化模型
        model = YoloDenoiser()
        model_path = f"/data/sdv3/xiajinsong/jiuzhou/denoiser/work_dir/yolo_deepnoise/yolo_deepDenoiser_20_1980.pth"
        # model = YoloDenoiser3feat()
        # model_path = f"/data/sdv3/xiajinsong/jiuzhou/denoiser/work_dir/yolo_3feat_deepnoise/yolo_deepDenoiser_20_1980.pth"
        model.cuda()
        foreground_list = []
        noiseWave_list = []
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # noiseSpec_t = torch.tensor(noiseSpec).cuda().unsqueeze(0).unsqueeze(0)
        ori_h, ori_w = noiseSpec.shape
        target_h, target_w = (ori_h - 1) // 32 + 1, (ori_w - 1) // 32 + 1
        new_noiseSpec = np.zeros((target_h * 32, target_w * 32), dtype=np.float32)
        new_noiseSpec[:ori_h, :ori_w] = noiseSpec.astype(np.float32)
        new_noiseSpec = torch.tensor(new_noiseSpec).cuda().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # → [3, H, W]
        new_noiseSpec = new_noiseSpec.float()
        print('new_noiseSpec', new_noiseSpec.shape)
        with torch.no_grad():
            res = model(new_noiseSpec, flag='train')
            sf = nn.Softmax(dim=1).cuda()
            res = sf(res)
            res = res.detach().cpu().numpy()[0]
            conf = res[0]
            conf = conf[:ori_h, :ori_w] 
            entroy_loss = - (mask * np.log(conf) + (1 - mask) * np.log(1 -conf))
            entroy_loss = np.mean(entroy_loss)
            conf[conf < thresh] = 0
    # 置信度后处理res[0]?不好卡 且结果杂乱
    foreground = conf * noiseSpec
    # background = res[1] * noiseSpec
    
    # 保存 .mat 文件
    io.savemat(os.path.join(output_folder, f"after_train.mat"), {
        f'res': res[0],
        f'before': noiseSpec,
        f'after': foreground
    })


    conf[conf >= thresh] = 1
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    conf = conf.astype(np.int32)
    mask = mask.astype(np.int32)
    print('mask.sum', mask.sum())
    iou = 0 if np.sum(conf | mask) == 0 else np.sum(conf & mask) / np.sum(conf | mask)
    print('==============iou==============', iou)
    print('==============entroy_loss==============', entroy_loss)
    # 保存频谱总和图
    plt.figure()
    plt.plot(noiseSpec[:, :].sum(0))
    plt.title('Before Denoising Spectrum Sum')
    plt.savefig(os.path.join(output_folder, f"before_stft_sum.png"))
    plt.close()

    plt.figure()
    plt.plot(foreground[:, :].sum(0))
    plt.title('After Denoising Spectrum Sum')
    plt.savefig(os.path.join(output_folder, f"after_stft_sum.png"))
    plt.close()
    
    return foreground, conf

flag = 'train'
output_folder = "z_output_fakedata_zhuanli/" + flag + "/images"
data = loadmat(output_folder + '/mask.mat')
noisy_spec = data['noisy']  # 结构体被当作 numpy object array
mask = data['tmp_mask']
denoise_noisy_spec, conf = test_deepdenoiser_stft_fragment(noisy_spec, mask)
denoise_image = to_image(denoise_noisy_spec)
cv2.imwrite(f"./{output_folder}/denoise_image.png", denoise_image)
conf_image = to_image(conf)
cv2.imwrite(f"./{output_folder}/conf_image.png", conf_image)
# res[res <= 0.5] = 0  # 卡阈值预处理 降低运算量
trace_img_vis, peaks_img, trajs = extract_multi_lines(denoise_noisy_spec, out_prefix='demo_multi',
                                prominence=5, min_length=40, delta_f=10, fre_range=10)
peaks_img = to_image(peaks_img)
trace_img_vis = to_image(trace_img_vis)
cv2.imwrite(f"./{output_folder}/peaks_img.png", peaks_img)
cv2.imwrite(f"./{output_folder}/trace_img_vis.png", trace_img_vis)