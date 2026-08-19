import argparse
import json
import os
import numpy as np
import cv2
from PIL import Image

from drift_sense.structures import GENERATORS
from drift_sense.degrade import degrade_pipeline

STYLE_PARAMS = {
    "dram": dict(pitch=48, line_width=6, via_radius=5),
    "finfet": dict(pitch=20, fin_width=5, gate_width=32),
}


def build_pair(style, seed, search_size=1000, ref_size=None):
    rng = np.random.default_rng(seed)

    downsample = rng.uniform(8.0, 10.5)
    fov = int(search_size * downsample)
    ref_size = ref_size or int(rng.uniform(900, 1100))

    params = dict(STYLE_PARAMS[style])
    params["size"] = fov
    params["seed"] = int(rng.integers(0, 2 ** 31))
    params["fid_len"] = int(ref_size * 0.16)
    params["fid_thickness"] = max(8, int(ref_size * 0.02))

    canvas, (tx, ty) = GENERATORS[style](**params)

    half = ref_size // 2
    margin = half + 40
    if tx < margin or tx > fov - margin or ty < margin or ty > fov - margin:
        tx = int(np.clip(tx, margin, fov - margin))
        ty = int(np.clip(ty, margin, fov - margin))

    reference = canvas[ty - half:ty + half, tx - half:tx + half].copy()

    ref_rng = np.random.default_rng(seed * 2 + 1)
    reference = degrade_pipeline(
        reference, ref_rng,
        blur_sigma=ref_rng.uniform(0.3, 0.8),
        rotation_deg=ref_rng.uniform(-1.5, 1.5),
        scale_jitter=1.0,
        poisson_scale=rng.uniform(6, 12),
        gaussian_sigma=rng.uniform(1.5, 3.5),
        edge_gain=ref_rng.uniform(0.35, 0.55),
    )

    search_full = cv2.resize(canvas, (search_size, search_size), interpolation=cv2.INTER_AREA)
    gt_x = tx * search_size / fov
    gt_y = ty * search_size / fov

    search_rng = np.random.default_rng(seed * 2 + 2)
    search = degrade_pipeline(
        search_full, search_rng,
        blur_sigma=search_rng.uniform(0.8, 1.8),
        rotation_deg=search_rng.uniform(-4, 4),
        scale_jitter=1.0,
        poisson_scale=rng.uniform(3, 7),
        gaussian_sigma=rng.uniform(4, 9),
        edge_gain=search_rng.uniform(0.45, 0.75),
    )

    return reference, search, (gt_x, gt_y), downsample, ref_size


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense synthetic dataset generator")
    parser.add_argument("--style", choices=["dram", "finfet", "both"], default="both")
    parser.add_argument("--num-pairs", type=int, default=30)
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ref_dir = os.path.join(args.output_dir, "reference")
    search_dir = os.path.join(args.output_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records = []

    for i in range(args.num_pairs):
        style = args.style if args.style != "both" else ("dram" if i % 2 == 0 else "finfet")
        seed = int(rng.integers(0, 2 ** 31))

        reference, search, (gt_x, gt_y), downsample, ref_size = build_pair(style, seed)

        ref_name = f"{style}_{i:03d}_reference.png"
        search_name = f"{style}_{i:03d}_search.png"
        Image.fromarray(reference).save(os.path.join(ref_dir, ref_name))
        Image.fromarray(search).save(os.path.join(search_dir, search_name))

        records.append({
            "id": i,
            "style": style,
            "reference_path": os.path.join("reference", ref_name),
            "search_path": os.path.join("search", search_name),
            "gt_x": round(gt_x, 2),
            "gt_y": round(gt_y, 2),
            "downsample_factor": round(downsample, 3),
            "reference_size": ref_size,
            "seed": seed,
        })

        print(f"[{i + 1}/{args.num_pairs}] {style:6s} -> gt=({gt_x:.1f}, {gt_y:.1f})  "
              f"downsample={downsample:.2f}x")

    gt_path = os.path.join(args.output_dir, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nGenerated {args.num_pairs} pairs in {args.output_dir}")
    print(f"Ground truth written to {gt_path}")


if __name__ == "__main__":
    main()
