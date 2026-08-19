import argparse
import json
import os
import time
import tracemalloc
import numpy as np
import cv2

from drift_sense.matcher import locate_reference


def main():
    parser = argparse.ArgumentParser(description="Runtime/memory/precision benchmark")
    parser.add_argument("--data-dir", default="./data_benchmark")
    parser.add_argument("--output", default="./results/benchmark.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from partial progress file (for chunked runs)")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "ground_truth.json")) as f:
        records = json.load(f)
    if args.limit:
        records = records[:args.limit]

    errors, times, peak_mems = [], [], []
    progress_path = args.output + ".progress"
    start_i = 0
    if args.resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            saved = json.load(f)
        errors, times, peak_mems = saved["errors"], saved["times"], saved["peak_mems"]
        start_i = len(errors)
        print(f"Resuming from {start_i}/{len(records)}")

    t_start = time.time()
    for i in range(start_i, len(records)):
        if args.resume and time.time() - t_start > 45:
            break
        r = records[i]
        ref = cv2.imread(os.path.join(args.data_dir, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(args.data_dir, r["search_path"]), cv2.IMREAD_GRAYSCALE)

        tracemalloc.start()
        t0 = time.time()
        result = locate_reference(ref, search)
        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        err = float(np.hypot(result.x - r["gt_x"], result.y - r["gt_y"]))
        errors.append(err)
        times.append(elapsed)
        peak_mems.append(peak / (1024 * 1024))

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            print(f"[{i + 1}/{len(records)}] running mean err={np.mean(errors):.2f}px "
                  f"avg time={np.mean(times):.3f}s")

    if args.resume and len(errors) < len(records):
        with open(progress_path, "w") as f:
            json.dump({"errors": errors, "times": times, "peak_mems": peak_mems}, f)
        print(f"Partial progress saved: {len(errors)}/{len(records)} — rerun with --resume to continue")
        return

    errors = np.array(errors)
    times = np.array(times)
    peak_mems = np.array(peak_mems)

    summary = {
        "n_pairs": len(records),
        "accuracy": {
            "mean_px": float(errors.mean()),
            "median_px": float(np.median(errors)),
            "std_px": float(errors.std()),
            "min_px": float(errors.min()),
            "max_px": float(errors.max()),
            "p50_px": float(np.percentile(errors, 50)),
            "p75_px": float(np.percentile(errors, 75)),
            "p90_px": float(np.percentile(errors, 90)),
            "p95_px": float(np.percentile(errors, 95)),
            "p99_px": float(np.percentile(errors, 99)),
            "within_15px_pct": float(100 * np.mean(errors < 15)),
            "within_30px_pct": float(100 * np.mean(errors < 30)),
            "within_50px_pct": float(100 * np.mean(errors < 50)),
            "catastrophic_fails_gt100px": int(np.sum(errors > 100)),
        },
        "runtime": {
            "mean_s": float(times.mean()),
            "median_s": float(np.median(times)),
            "p95_s": float(np.percentile(times, 95)),
            "min_s": float(times.min()),
            "max_s": float(times.max()),
        },
        "memory": {
            "mean_peak_mb": float(peak_mems.mean()),
            "max_peak_mb": float(peak_mems.max()),
        },
        "raw_errors": errors.tolist(),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- Benchmark summary ({len(records)} unseen pairs) ---")
    print(f"Accuracy: mean={summary['accuracy']['mean_px']:.2f}px  "
          f"median={summary['accuracy']['median_px']:.2f}px  "
          f"p95={summary['accuracy']['p95_px']:.2f}px  "
          f"fails(>100px)={summary['accuracy']['catastrophic_fails_gt100px']}/{len(records)}")
    print(f"Runtime:  mean={summary['runtime']['mean_s']:.3f}s  "
          f"median={summary['runtime']['median_s']:.3f}s  "
          f"p95={summary['runtime']['p95_s']:.3f}s")
    print(f"Memory:   mean peak={summary['memory']['mean_peak_mb']:.1f}MB  "
          f"max peak={summary['memory']['max_peak_mb']:.1f}MB")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
