import socket
import struct
import numpy as np
from scipy.io import wavfile
import time

# 配置参数
tcp_host = "127.0.0.1"
tcp_port = 18888
pid = "20221128_143237"
# pid = "test_64k"
data_root = f"jiuzhou_613data/{pid}"
send_secs = 1000   # 只发前100秒，覆盖第一个80s窗口触发点

# 读取WAV文件
print("[INFO] Loading WAV files...")
fs_pt, ptData = wavfile.read(f"{data_root}/{pid}_Pt.wav")
fs_vx, vxData = wavfile.read(f"{data_root}/{pid}_Vx.wav")
fs_vy, vyData = wavfile.read(f"{data_root}/{pid}_Vy.wav")

if not (fs_pt == fs_vx == fs_vy):
    raise ValueError("Sampling rate mismatch")

fs = fs_pt
min_len = min(len(ptData), len(vxData), len(vyData))
ptData = ptData[:min_len].astype(np.float32)
vxData = vxData[:min_len].astype(np.float32)
vyData = vyData[:min_len].astype(np.float32)

total_seconds = min(min_len // fs, send_secs)
print(f"[INFO] fs={fs}, total_seconds={total_seconds}")

# 连接服务器
print(f"[INFO] Connecting to {tcp_host}:{tcp_port}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((tcp_host, tcp_port))
print("[INFO] Connected")

try:
    for sec in range(total_seconds):
        start = sec * fs
        end = start + fs

        pt_1s = ptData[start:end]
        vx_1s = vxData[start:end]
        vy_1s = vyData[start:end]

        # 构建数据包
        data = struct.pack('>i', fs)  # 采样率（大端）
        data += struct.pack(f'>{fs}f', *pt_1s)  # P通道
        data += struct.pack(f'>{fs}f', *vx_1s)  # Vx通道
        data += struct.pack(f'>{fs}f', *vy_1s)  # Vy通道

        # 发送
        sock.sendall(data)
        print(f"[INFO] Sent second {sec+1}/{total_seconds}")

        time.sleep(1)  # 模拟实时发送

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
finally:
    sock.close()
    print("[INFO] Connection closed")
