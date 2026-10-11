"""Stand-in for the ESP32 — posts synthetic feature packets at 1 Hz.

Lets you develop and demo the Pi side before the firmware exists, and is the seed
of the `--replay` demo fallback (docs/11-realtime-pipeline.md).

The channel count is read from the server's /health so it always matches whatever
config/live_map.json says — no editing two files.

    python fake_node.py                          # steady healthy machine
    python fake_node.py --degrade                # health decays over 5 minutes
    python fake_node.py http://pi:8000/ingest --degrade --period 0.2
"""
import argparse
import json
import math
import random
import time
import urllib.error
import urllib.request


def get_features(base: str) -> list[str]:
    with urllib.request.urlopen(base + "/health", timeout=3) as r:
        return json.load(r)["features"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8000/ingest")
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--loop", action="store_true",
                    help="restart the --degrade ramp every --ramp-s seconds "
                         "instead of holding at failure")
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--ramp-s", type=float, default=300.0)
    ap.add_argument("--replay", metavar="UNIT.npz",
                    help="stream a real cached unit from data/interim — the only "
                         "mode whose feature distribution matches training")
    a = ap.parse_args()

    base = a.url.rsplit("/", 1)[0]
    names = get_features(base)
    print(f"{len(names)} channels: {names}")

    replay = None
    if a.replay:
        import numpy as np
        z = np.load(a.replay)
        replay = z["signals"]
        if replay.shape[1] != len(names):
            # Attempt to dynamically slice channels by matching sensor IDs
            matched = False
            if "sensor_ids" in z:
                from pathlib import Path
                vocab_path = (Path(__file__).parent / "artifacts" / "sensor_vocab.json"
                              if (Path(__file__).parent / "artifacts" / "sensor_vocab.json").exists()
                              else Path(__file__).resolve().parents[1] / "config" / "sensor_vocab.json")
                if vocab_path.exists():
                    vocab = json.loads(vocab_path.read_text())
                    vocab = {k: v for k, v in vocab.items() if not k.startswith("_")}
                    req_ids = [vocab[n] for n in names if n in vocab]
                    file_ids = list(z["sensor_ids"])
                    if len(req_ids) == len(names) and all(rid in file_ids for rid in req_ids):
                        col_indices = [file_ids.index(rid) for rid in req_ids]
                        replay = replay[:, col_indices]
                        print(f"dynamically matched {len(col_indices)} channels from {a.replay}")
                        matched = True
            if not matched:
                raise SystemExit(f"{a.replay} has {replay.shape[1]} channels, "
                                 f"server expects {len(names)} — check config/live_map.json")
        print(f"replaying {a.replay}: {replay.shape[0]} timesteps to failure")

    t0 = time.time()
    seq = 0
    while True:
        if replay is not None:
            if seq >= len(replay):
                print("replay finished (unit reached end of life)")
                return
            feat = replay[seq].tolist()
        else:
            # Invented values. Useful for exercising the plumbing, but the
            # distribution is NOT the one the model trained on, so the predicted
            # health will be arbitrary. Use --replay for a coherent demo.
            if a.degrade:
                elapsed = (time.time() - t0) / a.ramp_s
                # --loop wraps the ramp so the demo runs forever: healthy ->
                # failing -> healthy again. Without it the ramp holds at failure.
                age = (elapsed % 1.0) if a.loop else min(elapsed, 1.0)
            else:
                age = 0.0
            feat = []
            for i, n in enumerate(names):
                base_val = 0.04 + 0.02 * (i % 3)
                if "kurt" in n:
                    # Kurtosis is non-monotonic: ~3 healthy, spikes at early
                    # spalling, then falls back toward 3 as the defect spreads.
                    v = 3.0 + 8 * math.exp(-((age - 0.45) ** 2) / 0.02)
                elif "crest" in n:
                    v = 3.5 + 1.5 * age
                else:                               # rms — grows roughly exponentially
                    v = base_val * (1 + 4 * age ** 2)
                feat.append(v + random.gauss(0, 0.02 * abs(v) + 1e-3))

        body = json.dumps({"node": "fake", "seq": seq,
                           "ts_ms": int((time.time() - t0) * 1000),
                           "clip": 0, "jitter_us": 0,
                           "feat": [round(v, 6) for v in feat]})
        req = urllib.request.Request(a.url, data=body.encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2) as r:
                print(seq, r.read().decode())
            seq += 1
        except urllib.error.URLError as e:
            print("post failed:", e)
        time.sleep(a.period)


if __name__ == "__main__":
    main()
