import argparse
import json
import os
import socket
import struct
import sys
import time

import numpy as np
from scipy.io import wavfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtt_api.worker_tcp import read_frame, write_frame


def load_triplet(data_root, pid):
    fs_pt, pt = wavfile.read(os.path.join(data_root, "%s_Pt.wav" % pid))
    fs_vx, vx = wavfile.read(os.path.join(data_root, "%s_Vx.wav" % pid))
    fs_vy, vy = wavfile.read(os.path.join(data_root, "%s_Vy.wav" % pid))
    if not (fs_pt == fs_vx == fs_vy):
        raise RuntimeError("Pt/Vx/Vy sampling rate mismatch")
    min_len = min(len(pt), len(vx), len(vy))
    return (
        fs_pt,
        np.asarray(pt[:min_len], dtype=np.float32),
        np.asarray(vx[:min_len], dtype=np.float32),
        np.asarray(vy[:min_len], dtype=np.float32),
    )


def make_payload(pt, vx, vy):
    return b"".join([
        np.ascontiguousarray(pt, dtype="<f4").tobytes(),
        np.ascontiguousarray(vx, dtype="<f4").tobytes(),
        np.ascontiguousarray(vy, dtype="<f4").tobytes(),
    ])


def drain_responses(sock, expected_seq):
    while True:
        header, payload = read_frame(sock)
        print("[RX]", header)
        if header.get("type") == "processing_result":
            data = json.loads(payload.decode("utf-8"))
            print("[RESULT] buoy=%s rel=%s->%s points=%d clusters=%d" % (
                data["buoy_id"],
                data["relative_start_time"],
                data["relative_end_time"],
                len(data["idae"]["line_points"]),
                len(data["recognition"]["clusters"]),
            ))
            return
        if header.get("type") in ("no_result", "error") and header.get("seq") == expected_seq:
            return


def main():
    parser = argparse.ArgumentParser(description="Send framed audio chunks to mtt_api.worker_tcp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--buoy", default="demo-buoy")
    parser.add_argument("--pid", default="20221128_143237")
    parser.add_argument("--data-root", default=os.path.join(ROOT, "jiuzhou_613data", "20221128_143237"))
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    fs, pt, vx, vy = load_triplet(args.data_root, args.pid)
    total_seconds = min(args.seconds, len(pt) // fs, len(vx) // fs, len(vy) // fs)

    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        sock.settimeout(None)
        write_frame(sock, {"type": "hello", "seq": 0, "client": "examples/send_tcp_audio.py"})
        print("[RX]", read_frame(sock)[0])

        for sec in range(total_seconds):
            start = sec * fs
            end = start + fs
            payload = make_payload(pt[start:end], vx[start:end], vy[start:end])
            write_frame(sock, {
                "type": "audio_chunk",
                "seq": sec + 1,
                "buoy_id": args.buoy,
                "fs": fs,
                "start_time": float(sec),
                "samples": fs,
                "channels": ["pt", "vx", "vy"],
                "dtype": "float32",
                "byte_order": "little",
            }, payload)
            drain_responses(sock, sec + 1)
            if args.realtime:
                time.sleep(1)


if __name__ == "__main__":
    main()
