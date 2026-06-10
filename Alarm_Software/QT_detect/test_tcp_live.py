"""
端到端 TCP 测试：起服务 → 发 90 秒音频 → 观察响应
用法：python test_tcp_live.py
"""
import json, os, socket, struct, subprocess, sys, time, threading
import numpy as np
from scipy.io import wavfile

ROOT   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
HOST, PORT = "127.0.0.1", 18889
DATA_ROOT  = os.path.join(ROOT, "jiuzhou_613data", "20221128_143237")
PID        = "20221128_143237"
SEND_SECS  = 90   # 发 90 秒，覆盖第一个 80s 窗口触发点

sys.path.insert(0, ROOT)
from mtt_api.worker_tcp import read_frame, write_frame


# ── 启动服务端子进程 ──────────────────────────────────────────
print("[TEST] 启动 mtt_api.worker_tcp 服务端...")
server_proc = subprocess.Popen(
    [PYTHON, "-m", "mtt_api.worker_tcp",
     "--host", HOST, "--port", str(PORT), "--device", "cpu"],
    cwd=ROOT,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)

# 收集服务端输出（后台线程）
server_lines = []
server_ready = threading.Event()

def _read_server():
    for line in server_proc.stdout:
        line = line.rstrip()
        server_lines.append(line)
        print("[SRV]", line)
        if "listening on" in line:
            server_ready.set()

threading.Thread(target=_read_server, daemon=True).start()

if not server_ready.wait(timeout=60):
    print("[TEST] 服务端 60 秒内未就绪，退出")
    server_proc.terminate()
    sys.exit(1)

time.sleep(1)   # 等 worker 进程完全起来

# ── 加载音频 ──────────────────────────────────────────────────
print("[TEST] 加载音频...")
fs, pt = wavfile.read(os.path.join(DATA_ROOT, f"{PID}_Pt.wav"))
_,  vx = wavfile.read(os.path.join(DATA_ROOT, f"{PID}_Vx.wav"))
_,  vy = wavfile.read(os.path.join(DATA_ROOT, f"{PID}_Vy.wav"))
pt = pt.astype(np.float32)
vx = vx.astype(np.float32)
vy = vy.astype(np.float32)
total_secs = min(SEND_SECS, len(pt) // fs)
print(f"[TEST] fs={fs}  发送 {total_secs} 秒")

# ── 连接 & 发送 ───────────────────────────────────────────────
print(f"[TEST] 连接 {HOST}:{PORT}...")
sock = socket.create_connection((HOST, PORT), timeout=10)
sock.settimeout(None)

write_frame(sock, {"type": "hello", "seq": 0})
hdr, _ = read_frame(sock)
print("[RX ]", hdr)

errors   = []
results  = []
no_results = []

try:
    for sec in range(total_secs):
        s = sec * fs
        payload = b"".join([
            np.ascontiguousarray(pt[s:s+fs], dtype="<f4").tobytes(),
            np.ascontiguousarray(vx[s:s+fs], dtype="<f4").tobytes(),
            np.ascontiguousarray(vy[s:s+fs], dtype="<f4").tobytes(),
        ])
        write_frame(sock, {
            "type": "audio_chunk", "seq": sec+1,
            "buoy_id": "test-buoy", "fs": fs,
            "start_time": float(sec), "samples": fs,
            "channels": ["pt","vx","vy"],
            "dtype": "float32", "byte_order": "little",
        }, payload)

        # 读 ack
        ack_hdr, _ = read_frame(sock)
        # 读结果
        res_hdr, res_payload = read_frame(sock)
        t = res_hdr.get("type")
        seq = res_hdr.get("seq")

        if t == "processing_result":
            data = json.loads(res_payload.decode())
            pts  = len(data["idae"]["line_points"])
            results.append(seq)
            print("[RX ] sec=%3d  OK processing_result  line_points=%d" % (sec+1, pts))
        elif t == "no_result":
            no_results.append(seq)
            if (sec+1) % 10 == 0:
                print("[RX ] sec=%3d  -- no_result" % (sec+1,))
        elif t == "error":
            errors.append({"seq": seq, "sec": sec+1, "msg": res_hdr.get("message","")})
            print("[RX ] sec=%3d  ERR %s" % (sec+1, res_hdr.get("message","")[:120]))
        else:
            print("[RX ] sec=%3d  ??? %s" % (sec+1, res_hdr))

finally:
    sock.close()
    server_proc.terminate()
    server_proc.wait()

# ── 汇总 ─────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"发送总秒数   : {total_secs}")
print(f"no_result 次 : {len(no_results)}")
print(f"processing_result 次: {len(results)}")
print(f"error 次     : {len(errors)}")
if errors:
    print("\n错误详情：")
    for e in errors:
        print(f"  sec={e['sec']} seq={e['seq']}  {e['msg'][:200]}")
