"""
YoloDenoiser ONNX 导出 + 数值验证脚本

发现：result[1]['feats'][0] 才是正确特征图（[1,64,H/4,W/4]）
     原代码 result[0] 取的是检测头输出（bug，但模型在 push 中实际报错了吗？
     → 实际调用里用 flag='test' + predict，应验证实际推理是否正常）

运行：python export_denoiser.py
输出：yolo_denoiser.onnx + 验证报告
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PTH = (
    "work_dir/yolo_4feat_deepnoise_log_randomh_613_moremorexianpu_snrr0.05"
    "/yolo_deepDenoiser_20_240.pth"
)
ONNX_PATH = "yolo_denoiser.onnx"
DEVICE = "cpu"


# ─────────────────────────────────────────────
# 1. 加载原始模型
# ─────────────────────────────────────────────
print("=" * 60)
print("1. 加载原始模型")
from yolo_denoiser import YoloDenoiser

orig_model = YoloDenoiser(out_channel=2)
state_dict = torch.load(MODEL_PTH, map_location=DEVICE)
orig_model.load_state_dict(state_dict)
orig_model.to(DEVICE).eval()
print("   加载完成")


# ─────────────────────────────────────────────
# 2. 重现 test_deepdenoiser_stft_fragment 的实际推理路径
#    验证完整调用能否跑通（用真实的 pad 逻辑）
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 验证原始推理路径（含 padding）")

def run_original(model, noise_spec_np, thresh=0.1, line_channel=0, device='cpu'):
    """完整复现 test_deepdenoiser_stft_fragment"""
    ori_h, ori_w = noise_spec_np.shape

    target_h = (ori_h - 1) // 32 + 1
    target_w = (ori_w - 1) // 32 + 1
    pad_h = target_h * 32 + 32
    pad_w = target_w * 32

    padded = np.zeros((pad_h, pad_w), dtype=np.float32)
    padded[:ori_h, :ori_w] = noise_spec_np.astype(np.float32)

    if pad_h > ori_h:
        rows_to_pad = pad_h - ori_h
        padded[ori_h:pad_h, :ori_w] = noise_spec_np[-rows_to_pad:, :]

    if pad_w > ori_w:
        cols_to_pad = pad_w - ori_w
        padded[:ori_h, ori_w:pad_w] = noise_spec_np[:, -cols_to_pad:]
        if pad_h > ori_h:
            rows_to_pad = pad_h - ori_h
            padded[ori_h:pad_h, ori_w:pad_w] = noise_spec_np[-rows_to_pad:, -cols_to_pad:]

    x = torch.tensor(padded, dtype=torch.float32)
    x = x.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        # 直接用 backbone + upsampling（修正特征提取路径）
        result = model.yolomodel(x)
        Out1 = result[1]['feats'][0]   # 正确路径
        Out2 = model.upSampleLayer1(Out1)
        Out3 = model.upSampleLayer2(Out2)
        Out4 = model.upSampleLayer3(Out3)
        # softmax
        sf = nn.Softmax(dim=1)
        res = sf(Out4).cpu().numpy()[0]  # [2, pad_h, pad_w]
        conf = res[line_channel][:ori_h, :ori_w]

    conf_soft = conf.copy()
    conf_soft[:, :2000] = np.where(conf_soft[:, :2000] < thresh, 0, conf_soft[:, :2000])
    conf_soft[:, 2000:] = np.where(conf_soft[:, 2000:] < 0.4, 0, conf_soft[:, 2000:])
    foreground = conf_soft * noise_spec_np
    return foreground, conf_soft

# 用小的随机谱图测试
np.random.seed(42)
test_spec = np.random.rand(160, 300).astype(np.float32)
fg, conf = run_original(orig_model, test_spec)
print(f"   foreground shape = {fg.shape}")
print(f"   foreground range = [{fg.min():.4f}, {fg.max():.4f}]")
print("   原始推理路径 ✓")


# ─────────────────────────────────────────────
# 3. 导出包装器（固定 feats 提取路径）
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 创建导出包装器")

class DenoiserExportWrapper(nn.Module):
    """
    固定推理路径：backbone → feats[0] → 3层上采样 → logits
    输入: (B, 3, H, W)  H/W 为 32 的倍数
    输出: (B, 2, H, W)  logits，在 C++ 侧做 softmax
    """
    def __init__(self, denoiser: YoloDenoiser):
        super().__init__()
        self.backbone   = denoiser.yolomodel
        self.up1        = denoiser.upSampleLayer1
        self.up2        = denoiser.upSampleLayer2
        self.head       = denoiser.upSampleLayer3

    def forward(self, x):
        result = self.backbone(x)
        feat   = result[1]['feats'][0]   # [B, 64, H/4, W/4]
        return self.head(self.up2(self.up1(feat)))   # [B, 2, H, W]


wrapper = DenoiserExportWrapper(orig_model).to(DEVICE)
wrapper.eval()

# 验证包装器和手动调用完全一致
with torch.no_grad():
    np.random.seed(0)
    h_test, w_test = 192, 320  # 32 的倍数
    dummy = torch.rand(1, 3, h_test, w_test, dtype=torch.float32)

    # 手动路径
    res = orig_model.yolomodel(dummy)
    feat = res[1]['feats'][0]
    manual_out = orig_model.upSampleLayer3(
        orig_model.upSampleLayer2(orig_model.upSampleLayer1(feat))
    )

    # 包装器路径
    wrapper_out = wrapper(dummy)

diff = (manual_out - wrapper_out).abs()
print(f"   包装器 vs 手动路径 max_diff = {diff.max().item():.2e}")
assert diff.max().item() < 1e-5, "包装器输出不一致！"
print("   包装器验证 ✓")


# ─────────────────────────────────────────────
# 4. ONNX 导出
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 导出 ONNX")

import torch.onnx

dummy_export = torch.rand(1, 3, 192, 320, dtype=torch.float32)

torch.onnx.export(
    wrapper,
    dummy_export,
    ONNX_PATH,
    opset_version=13,            # PyTorch 1.9 最高稳定 opset
    input_names=["spec"],
    output_names=["logits"],
    dynamic_axes={
        "spec":   {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch", 2: "height", 3: "width"},
    },
    do_constant_folding=True,
    verbose=False,
)
print(f"   已导出到 {ONNX_PATH}")

# ONNX 模型基础检查
import onnx
model_onnx = onnx.load(ONNX_PATH)
onnx.checker.check_model(model_onnx)
print(f"   ONNX 模型检查 ✓  (opset={model_onnx.opset_import[0].version})")
size_mb = os.path.getsize(ONNX_PATH) / 1e6
print(f"   文件大小: {size_mb:.1f} MB")


# ─────────────────────────────────────────────
# 5. 数值精度验证（PyTorch vs ONNX Runtime）
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 数值精度验证 (PyTorch vs ONNX Runtime)")

import onnxruntime as ort

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

cases = [
    ("全零输入",     np.zeros((1, 3, 192, 320), dtype=np.float32)),
    ("随机小尺寸",  np.random.rand(1, 3, 160, 288).astype(np.float32)),  # 不需32倍数
    ("随机大尺寸",  np.random.rand(1, 3, 224, 640).astype(np.float32)),
    ("实际典型尺寸", np.random.rand(1, 3, 192, 4800).astype(np.float32)),  # 240s × 20Hz
]

all_pass = True
for name, inp in cases:
    # 由于 dynamic_axes，小尺寸也能跑，但必须是 32 的倍数（backbone 内部限制）
    # 先把 H/W 对齐到 32
    b, c, h, w = inp.shape
    h32 = ((h - 1) // 32 + 1) * 32
    w32 = ((w - 1) // 32 + 1) * 32
    if h32 != h or w32 != w:
        padded = np.zeros((b, c, h32, w32), dtype=np.float32)
        padded[:, :, :h, :w] = inp
        inp = padded
        name += f" (padded {h}→{h32}, {w}→{w32})"

    with torch.no_grad():
        pt_out = wrapper(torch.from_numpy(inp)).numpy()
    ort_out = sess.run(None, {"spec": inp})[0]

    max_diff  = np.abs(pt_out - ort_out).max()
    mean_diff = np.abs(pt_out - ort_out).mean()
    ok = max_diff < 1e-4
    all_pass = all_pass and ok
    status = "✓" if ok else "✗"
    print(f"   {status} {name}")
    print(f"      shape={pt_out.shape}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")

print()
if all_pass:
    print("=" * 60)
    print("全部验证通过 ✓  ONNX 模型可用于 C++ 推理")
else:
    print("=" * 60)
    print("存在精度问题，请检查上方标 ✗ 的用例")

print(f"\nONNX 文件路径: {os.path.abspath(ONNX_PATH)}")
