import os
import sys

from scipy.io import wavfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtt_api import AudioChunk, MarineAlgorithmPipeline, ProcessorConfig


def main():
    pid = "20221128_143237"
    data_root = os.path.join(ROOT, "synthetic_data")

    fs_pt, pt = wavfile.read(os.path.join(data_root, "%s_Pt.wav" % pid))
    fs_vx, vx = wavfile.read(os.path.join(data_root, "%s_Vx.wav" % pid))
    fs_vy, vy = wavfile.read(os.path.join(data_root, "%s_Vy.wav" % pid))

    if not (fs_pt == fs_vx == fs_vy):
        raise RuntimeError("Pt/Vx/Vy sampling rate mismatch")

    config = ProcessorConfig.default(ROOT, device="cpu")
    pipeline = MarineAlgorithmPipeline(config)

    fs = fs_pt
    seconds_per_push = 1
    samples_per_push = fs * seconds_per_push
    max_seconds = min(100, len(pt) // fs, len(vx) // fs, len(vy) // fs)

    last_result = None
    for sec in range(max_seconds):
        start = sec * samples_per_push
        end = start + samples_per_push
        chunk = AudioChunk.from_arrays(
            buoy_id="demo-buoy",
            fs=fs,
            start_time=0.0 + sec,
            pt=pt[start:end],
            vx=vx[start:end],
            vy=vy[start:end],
        )
        last_result = pipeline.push_audio(chunk, include_array_data=False)

    if last_result is None:
        print("no result")
        return

    data = last_result.to_dict()
    print("buoy:", data["buoy_id"])
    print("time:", data["relative_start_time"], "->", data["relative_end_time"])
    print("spec shape:", data["idae"]["cols"], "x", data["idae"]["rows"])
    print("line points:", len(data["idae"]["line_points"]))
    print("trajectories:", len(data["idae"]["trajectories"]))
    print("clusters:", len(data["recognition"]["clusters"]))


if __name__ == "__main__":
    main()
