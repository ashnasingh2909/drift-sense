import argparse
import os
import time
import numpy as np
import cv2

from drift_sense.structures import GENERATORS
from drift_sense.cnn import DriftSenseCNN

PATCH = 64
STYLES = ["dram", "finfet"]
STYLE_PARAMS = {
    "dram": dict(pitch=48, line_width=6, via_radius=5),
    "finfet": dict(pitch=20, fin_width=5, gate_width=32),
}


def sample_patch_pair(rng):
    style = STYLES[rng.integers(0, 2)]
    canvas_size = 1400
    params = dict(STYLE_PARAMS[style])
    params["size"] = canvas_size
    params["seed"] = int(rng.integers(0, 2 ** 31))
    params["fid_len"] = int(PATCH * 4 * 0.16)
    params["fid_thickness"] = max(6, int(PATCH * 4 * 0.02))
    canvas, (tx, ty) = GENERATORS[style](**params)

    half = PATCH * 2
    margin = half + 20
    tx = int(np.clip(tx, margin, canvas_size - margin))
    ty = int(np.clip(ty, margin, canvas_size - margin))

    anchor_big = canvas[ty - half:ty + half, tx - half:tx + half]
    anchor = cv2.resize(anchor_big, (PATCH, PATCH), interpolation=cv2.INTER_AREA)

    jitter = rng.integers(-4, 5, size=2)
    py = int(np.clip(ty + jitter[1], half, canvas_size - half))
    px = int(np.clip(tx + jitter[0], half, canvas_size - half))
    positive_big = canvas[py - half:py + half, px - half:px + half]
    positive = cv2.resize(positive_big, (PATCH, PATCH), interpolation=cv2.INTER_AREA)

    noise = rng.normal(0, 6, positive.shape)
    positive = np.clip(positive.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    off_x = int(rng.choice([-1, 1]) * rng.uniform(half * 1.5, canvas_size / 2 - half))
    off_y = int(rng.choice([-1, 1]) * rng.uniform(half * 1.5, canvas_size / 2 - half))
    nx = int(np.clip(tx + off_x, half, canvas_size - half))
    ny = int(np.clip(ty + off_y, half, canvas_size - half))
    negative_big = canvas[ny - half:ny + half, nx - half:nx + half]
    negative = cv2.resize(negative_big, (PATCH, PATCH), interpolation=cv2.INTER_AREA)

    return anchor, positive, negative


def to_tensor(img):
    return (img.astype(np.float32) / 255.0)[None, None, :, :]


def main():
    parser = argparse.ArgumentParser(description="Train the Drift-Sense CNN embedding model")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="./models/drift_sense_cnn.npz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    net = DriftSenseCNN(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    losses = []

    for it in range(args.iterations):
        anchor, positive, negative = sample_patch_pair(rng)
        a_t, p_t, n_t = to_tensor(anchor), to_tensor(positive), to_tensor(negative)

        feat_a = net.forward(a_t, train=True)
        pooled_a = feat_a.reshape(1, 16, -1).mean(axis=2)
        norm_a = np.linalg.norm(pooled_a, axis=1, keepdims=True) + 1e-8
        emb_a = pooled_a / norm_a

        feat_p = net.forward(p_t, train=True)
        pooled_p = feat_p.reshape(1, 16, -1).mean(axis=2)
        norm_p = np.linalg.norm(pooled_p, axis=1, keepdims=True) + 1e-8
        emb_p = pooled_p / norm_p

        feat_n = net.forward(n_t, train=True)
        pooled_n = feat_n.reshape(1, 16, -1).mean(axis=2)
        norm_n = np.linalg.norm(pooled_n, axis=1, keepdims=True) + 1e-8
        emb_n = pooled_n / norm_n

        pos_sim = float((emb_a * emb_p).sum())
        neg_sim = float((emb_a * emb_n).sum())
        margin = 0.2
        pos_loss = (1 - pos_sim) ** 2
        neg_loss = max(0.0, neg_sim + margin) ** 2
        loss = pos_loss + neg_loss
        losses.append(loss)

        dsim_pos = -2 * (1 - pos_sim)
        dsim_neg = 2 * max(0.0, neg_sim + margin)

        demb_a = dsim_pos * emb_p + dsim_neg * emb_n
        demb_p = dsim_pos * emb_a
        demb_n = dsim_neg * emb_a

        dpooled_a = demb_a / norm_a
        dpooled_p = demb_p / norm_p
        dpooled_n = demb_n / norm_n

        net.backward_from_pooled_grad(dpooled_p, args.lr)
        net.backward_from_pooled_grad(dpooled_n, args.lr)
        net.backward_from_pooled_grad(dpooled_a, args.lr)

        if it % 50 == 0 or it == args.iterations - 1:
            recent = np.mean(losses[-50:])
            print(f"[{it:4d}/{args.iterations}] loss={loss:.4f} "
                  f"avg50={recent:.4f} pos_sim={pos_sim:.3f} neg_sim={neg_sim:.3f}")

    net.save(args.output)
    print(f"\nTrained in {time.time() - t0:.1f}s over {args.iterations} iterations")
    print(f"Initial avg loss (first 50): {np.mean(losses[:50]):.4f}")
    print(f"Final avg loss (last 50):    {np.mean(losses[-50:]):.4f}")
    print(f"Weights saved to {args.output}")


if __name__ == "__main__":
    main()
