import os
import glob
import numpy as np
import soundfile as sf   # pip install soundfile

# === 1. 用户可改的两处路径 ===
src_dir = '/data/sdv3/xiangrui/shipsear_5s_16k/1/1_0'                       # 源文件夹
out_dir = '/data/sdv3/xiangrui/shipsear_5s_16k/1/1_0'             # 新文件夹（可任意命名）
out_file = os.path.join(out_dir, 'merged.wav')

# === 2. 创建输出目录 ===
os.makedirs(out_dir, exist_ok=True)

# === 3. 按编号排序获取文件列表 ===
pattern = os.path.join(src_dir, '1_0_*.wav')
files = sorted(glob.glob(pattern),
               key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]))

if not files:
    raise RuntimeError(f'在 {src_dir} 下没有找到匹配文件！')

# === 4. 拼接音频 ===
audio_all = []
for f in files:
    data, sr = sf.read(f)
    if sr != 16000:
        raise ValueError(f'{f} 采样率不是 16000！')
    audio_all.append(data)

merged = np.concatenate(audio_all)

# === 5. 保存结果 ===
sf.write(out_file, merged, 16000)
print(f'已生成 {out_file}，总长度 {len(merged)/16000:.2f} 秒')