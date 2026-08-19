import argparse
import json
import os
import time
import numpy as np
import cv2

from drift_sense.matcher import locate_reference_ablation

STAGES = [
    ("1. NCC only (fixed scale/angle)",
     dict(use_multiscale=False, use_multiangle=False, use_voting=False, use_periodicity=False)),
    ("2. + multi-scale",
     dict(use_multiscale=True, use_multiangle=False, use_voting=False, use_periodicity=False)),
    ("3. + multi-scale + multi-angle",
     dict(use_multiscale=True, use_multiangle=True, use_voting=False, use_periodicity=False)),
    ("4. + voting (no periodicity tie-break)",
     dict(use_multiscale=True, use_multiangle=True, use_voting=True, use_periodicity=False)),
    ("5. + periodicity-aware disambiguation (full)",
     dict(use_multiscale=True, use_multiangle=True, use_voting=True, use_periodicity=True)),
]


def main():
    parser = argparse.ArgumentParser(description="Ablation study across matcher stages")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of pairs (speed)")
    parser.add_argument("--stages", default=None,
                         help="Comma-separated stage numbers to run, e.g. '1,2,3' (default: all)")
    parser.add_argument("--merge", action="store_true",
                         help="Merge with existing ./results/ablation.json instead of overwriting")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "ground_truth.json")) as f:
        records = json.load(f)
    if args.limit:
        records = records[:args.limit]

    stages_to_run = STAGES
    if args.stages:
        wanted = set(int(s) for s in args.stages.split(","))
        stages_to_run = [(name, kw) for i, (name, kw) in enumerate(STAGES, 1) if i in wanted]

    print(f"Running ablation on {len(records)} pairs across {len(stages_to_run)} configurations\n")

    results = {}
    if args.merge and os.path.exists("./results/ablation.json"):
        with open("./results/ablation.json") as f:
            results = json.load(f)

    for name, kwargs in stages_to_run:
        errors = []
        t0 = time.time()
        for r in records:
            ref = cv2.imread(os.path.join(args.data_dir, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
            search = cv2.imread(os.path.join(args.data_dir, r["search_path"]), cv2.IMREAD_GRAYSCALE)
            res = locate_reference_ablation(ref, search, **kwargs)
            err = float(np.hypot(res.x - r["gt_x"], res.y - r["gt_y"]))
            errors.append(err)
        elapsed = time.time() - t0
        errors = np.array(errors)
        results[name] = {
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "fails_100px": int(np.sum(errors > 100)),
            "avg_time_ms": 1000 * elapsed / len(records),
        }
        print(f"{name:48s} mean={results[name]['mean']:7.2f}px  "
              f"median={results[name]['median']:7.2f}px  "
              f"p90={results[name]['p90']:7.2f}px  "
              f"fails(>100px)={results[name]['fails_100px']:3d}/{len(records)}  "
              f"avg={results[name]['avg_time_ms']:6.1f}ms/pair")

    os.makedirs("./results", exist_ok=True)
    with open("./results/ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWritten to ./results/ablation.json")


if __name__ == "__main__":
    main()
