import json
import os
import sys

from scipy.io import wavfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtt_api import AudioChunk, MarineAlgorithmPipeline, ProcessorConfig


def trim_result(result):
    data = result.to_dict()
    idae = data["idae"]

    # Keep this file readable: matrices retain shape/dtype only, and list
    # fields include a small representative prefix.
    for key in ("power_db", "doa_matrix", "noisy_spec", "denoise_spec", "time_azimuth"):
        payload = idae.get(key)
        if payload is not None:
            payload["data"] = None

    idae["freq_axis"] = idae["freq_axis"][:10]
    idae["time_axis"] = idae["time_axis"][:10]
    idae["line_points"] = idae["line_points"][:8]

    trimmed_trajectories = []
    for traj in idae["trajectories"][:3]:
        trimmed_trajectories.append({
            "track_id": traj["track_id"],
            "times": traj["times"][:8],
            "frequencies": traj["frequencies"][:8],
            "doas": traj["doas"][:8],
            "total_points": len(traj["times"]),
        })
    idae["trajectories"] = trimmed_trajectories

    return data


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
    samples_per_push = fs
    max_seconds = min(100, len(pt) // fs, len(vx) // fs, len(vy) // fs)

    last_result = None
    for sec in range(max_seconds):
        start = sec * samples_per_push
        end = start + samples_per_push
        chunk = AudioChunk.from_arrays(
            buoy_id="demo-buoy",
            fs=fs,
            start_time=float(sec),
            pt=pt[start:end],
            vx=vx[start:end],
            vy=vy[start:end],
        )
        last_result = pipeline.push_audio(chunk, include_array_data=False)

    if last_result is None:
        raise RuntimeError("no result generated")

    out_dir = os.path.join(ROOT, "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "api_sample_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trim_result(last_result), f, ensure_ascii=False, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()
