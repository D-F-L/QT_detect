"""
第一步：探测 YoloDenoiser 的输出结构。
运行方式：从项目根目录执行
    python probe_model.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo_denoiser import YoloDenoiser

MODEL_PTH = (
    "work_dir/yolo_4feat_deepnoise_log_randomh_613_moremorexianpu_snrr0.05"
    "/yolo_deepDenoiser_20_240.pth"
)
DEVICE = "cpu"

print("=" * 60)
print("加载模型...")
model = YoloDenoiser(out_channel=2)
state_dict = torch.load(MODEL_PTH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()
print("模型加载成功")

# 用一个典型尺寸的 dummy 输入
# test_deepdenoiser_stft_fragment 里的 pad 逻辑保证输入是 32 的倍数
# 典型谱图 ~(160, 4000) → pad → (192, 4000)
H, W = 192, 320   # 先用小尺寸测试
dummy = torch.zeros(1, 3, H, W, dtype=torch.float32, device=DEVICE)

print("\n" + "=" * 60)
print("测试 forward() 路径 (flag='train')...")
with torch.no_grad():
    result_train = model.yolomodel(dummy)

print(f"  type(result_train) = {type(result_train)}")
if isinstance(result_train, dict):
    print(f"  keys: {list(result_train.keys())}")
    for k, v in result_train.items():
        if isinstance(v, (list, tuple)):
            print(f"    {k}: list/tuple len={len(v)}, [0].shape={v[0].shape}")
        elif isinstance(v, torch.Tensor):
            print(f"    {k}: Tensor shape={v.shape}")
elif isinstance(result_train, (list, tuple)):
    print(f"  list/tuple len={len(result_train)}")
    for i, v in enumerate(result_train):
        if isinstance(v, torch.Tensor):
            print(f"    [{i}]: shape={v.shape}")
        elif isinstance(v, (list, tuple)):
            print(f"    [{i}]: nested list/tuple len={len(v)}")
            for j, vv in enumerate(v):
                if isinstance(vv, torch.Tensor):
                    print(f"      [{j}]: shape={vv.shape}")
elif isinstance(result_train, torch.Tensor):
    print(f"  Tensor shape={result_train.shape}")

print("\n" + "=" * 60)
print("测试 predict() 路径 (flag='test')...")
with torch.no_grad():
    result_test = model.yolomodel.predict(dummy)

print(f"  type(result_test) = {type(result_test)}")
if isinstance(result_test, dict):
    print(f"  keys: {list(result_test.keys())}")
    for k, v in result_test.items():
        if isinstance(v, (list, tuple)):
            print(f"    {k}: list/tuple len={len(v)}, [0].shape={v[0].shape}")
        elif isinstance(v, torch.Tensor):
            print(f"    {k}: Tensor shape={v.shape}")
elif isinstance(result_test, (list, tuple)):
    print(f"  list/tuple len={len(result_test)}")
    for i, v in enumerate(result_test):
        if isinstance(v, torch.Tensor):
            print(f"    [{i}]: shape={v.shape}")
        elif isinstance(v, (list, tuple)):
            print(f"    [{i}]: nested list/tuple len={len(v)}")
            for j, vv in enumerate(v):
                if isinstance(vv, torch.Tensor):
                    print(f"      [{j}]: shape={vv.shape}")
elif isinstance(result_test, torch.Tensor):
    print(f"  Tensor shape={result_test.shape}")

print("\n" + "=" * 60)
print("详细打印 result 全部元素...")

def print_structure(val, prefix="", depth=0):
    if depth > 4:
        print(prefix + "...")
        return
    if isinstance(val, torch.Tensor):
        print(f"{prefix}Tensor shape={val.shape} dtype={val.dtype} range=[{val.min():.3f},{val.max():.3f}]")
    elif isinstance(val, dict):
        print(f"{prefix}dict keys={list(val.keys())}")
        for k, v in val.items():
            print_structure(v, prefix + f"  [{k}] ", depth + 1)
    elif isinstance(val, (list, tuple)):
        print(f"{prefix}{type(val).__name__} len={len(val)}")
        for i, v in enumerate(val):
            print_structure(v, prefix + f"  [{i}] ", depth + 1)
    else:
        print(f"{prefix}{type(val).__name__} = {val}")

with torch.no_grad():
    result_full = model.yolomodel(dummy)
print_structure(result_full, "result_train: ")

print()
with torch.no_grad():
    result_full2 = model.yolomodel.predict(dummy)
print_structure(result_full2, "result_test:  ")

print("\n" + "=" * 60)
print("寻找 shape=[1,64,H,W] 的特征图（upSampleLayer1 期望输入 64ch）...")

def find_tensor_by_channels(val, target_ch, path="result"):
    found = []
    if isinstance(val, torch.Tensor):
        if val.ndim == 4 and val.shape[1] == target_ch:
            found.append((path, val.shape))
    elif isinstance(val, (list, tuple)):
        for i, v in enumerate(val):
            found.extend(find_tensor_by_channels(v, target_ch, f"{path}[{i}]"))
    elif isinstance(val, dict):
        for k, v in val.items():
            found.extend(find_tensor_by_channels(v, target_ch, f"{path}['{k}']"))
    return found

with torch.no_grad():
    res = model.yolomodel(dummy)

for ch in [64, 128, 256]:
    hits = find_tensor_by_channels(res, ch, "result")
    if hits:
        print(f"  {ch}ch: {hits}")
    else:
        print(f"  {ch}ch: 未找到")

print("\n" + "=" * 60)
print("检查 upSampleLayer 实际权重形状...")
print(f"  upSampleLayer1.conv.weight.shape = {model.upSampleLayer1.conv.weight.shape}")
print(f"  upSampleLayer2.conv.weight.shape = {model.upSampleLayer2.conv.weight.shape}")
print(f"  upSampleLayer3.weight.shape      = {model.upSampleLayer3.weight.shape}")
print("探测完成。")
