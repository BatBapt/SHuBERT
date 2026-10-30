#!/usr/bin/env python3
"""Compare SHuBERT base vs adapted features, layer by layer.

    python compare_shubert.py --base-dir features/shubert_base --adapted-dir features/shubert_adapted \
        --n-videos 5 --out compare_shubert

Reads <video_id>.npy of shape (12, T, 768) in both folders, samples the same frames
in both, and reports per layer:
    scale     mean |x| and std, base and adapted
    cosine    mean frame-wise cosine similarity between base and adapted
    cka       linear CKA between base and adapted (scale-invariant similarity, 1 = same structure)
    cka_null  CKA after shuffling the adapted frames (chance level)
plus a 12 x 12 cross-layer CKA matrix (base layer i vs adapted layer j).

Writes <out>.csv and <out>.png. CPU only.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def linear_cka(x, y):
    """Linear CKA between (n, d) matrices; columns are centred inside."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = np.linalg.norm(x.T @ y) ** 2
    xtx = np.linalg.norm(x.T @ x)
    yty = np.linalg.norm(y.T @ y)
    return float(xty / (xtx * yty))


def load_frames(base_dir, adapted_dir, videos, max_frames, rng):
    """Return base, adapted arrays of shape (L, N, D) on the same sampled frames."""
    base, adapted = [], []
    for vid in videos:
        a = np.load(Path(base_dir) / f"{vid}.npy", mmap_mode="r")
        b = np.load(Path(adapted_dir) / f"{vid}.npy", mmap_mode="r")
        if a.shape != b.shape:
            raise ValueError(f"{vid}: shapes differ {a.shape} vs {b.shape}")
        t = a.shape[1]
        idx = np.sort(rng.choice(t, size=min(max_frames, t), replace=False))
        base.append(np.asarray(a[:, idx, :], dtype=np.float32))
        adapted.append(np.asarray(b[:, idx, :], dtype=np.float32))
        print(f"[load] {vid}: {a.shape}, {len(idx)} frames sampled")
    return np.concatenate(base, axis=1), np.concatenate(adapted, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--adapted-dir", required=True)
    p.add_argument("--videos", nargs="*", default=None, help="video ids (default: first --n-videos in common)")
    p.add_argument("--n-videos", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=4000, help="frames sampled per video")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="compare_shubert")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.videos:
        videos = args.videos
    else:
        common = sorted({f.stem for f in Path(args.base_dir).glob("*.npy")} &
                        {f.stem for f in Path(args.adapted_dir).glob("*.npy")})
        videos = common[:args.n_videos]
    if not videos:
        raise SystemExit("no common .npy files between the two folders")

    A, B = load_frames(args.base_dir, args.adapted_dir, videos, args.max_frames, rng)
    n_layers, n, _ = A.shape
    print(f"[compare] {len(videos)} videos, {n} frames, {n_layers} layers")

    rows = []
    perm = rng.permutation(n)
    for l in range(n_layers):
        a, b = A[l], B[l]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
        rows.append({
            "layer": l + 1,
            "base_mean_abs": float(np.abs(a).mean()), "adapted_mean_abs": float(np.abs(b).mean()),
            "base_std": float(a.std()), "adapted_std": float(b.std()),
            "cosine_mean": float(cos.mean()),
            "cka": linear_cka(a, b),
            "cka_null": linear_cka(a, b[perm]),
        })

    cross = np.array([[linear_cka(A[i], B[j]) for j in range(n_layers)] for i in range(n_layers)])

    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savetxt(f"{args.out}_cross_cka.csv", cross, delimiter=",", fmt="%.4f")

    print(f"\n{'layer':>5} {'|x| base':>10} {'|x| adapt':>10} {'cosine':>8} {'CKA':>7} {'CKA null':>9}")
    for r in rows:
        print(f"{r['layer']:>5} {r['base_mean_abs']:>10.4f} {r['adapted_mean_abs']:>10.4f} "
              f"{r['cosine_mean']:>8.3f} {r['cka']:>7.3f} {r['cka_null']:>9.3f}")
    best = cross.argmax(axis=0) + 1
    print("\n[cross] for each adapted layer, the most similar base layer:")
    print("  adapted: " + " ".join(f"{j:>3}" for j in range(1, n_layers + 1)))
    print("  base:    " + " ".join(f"{i:>3}" for i in best))

    L = np.arange(1, n_layers + 1)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    ax[0].plot(L, [r["base_mean_abs"] for r in rows], "o-", label="base")
    ax[0].plot(L, [r["adapted_mean_abs"] for r in rows], "o-", label="adapted")
    ax[0].set_title("mean |x| per layer"); ax[0].legend()
    ax[1].plot(L, [r["cosine_mean"] for r in rows], "o-")
    ax[1].set_title("frame-wise cosine, base vs adapted"); ax[1].set_ylim(-0.1, 1.05)
    ax[2].plot(L, [r["cka"] for r in rows], "o-", label="CKA")
    ax[2].plot(L, [r["cka_null"] for r in rows], "--", color="grey", label="shuffled (chance)")
    ax[2].set_title("linear CKA, base vs adapted"); ax[2].set_ylim(-0.05, 1.05); ax[2].legend()
    im = ax[3].imshow(cross, vmin=0, vmax=1, cmap="viridis", origin="lower")
    ax[3].set_xticks(range(n_layers)); ax[3].set_xticklabels(L)
    ax[3].set_yticks(range(n_layers)); ax[3].set_yticklabels(L)
    ax[3].set_xlabel("adapted layer"); ax[3].set_ylabel("base layer"); ax[3].set_title("cross-layer CKA")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a_ in ax[:3]:
        a_.set_xlabel("layer"); a_.set_xticks(L)
    fig.suptitle(f"SHuBERT base vs adapted ({len(videos)} videos, {n} frames)")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=120)
    print(f"\n[compare] -> {args.out}.csv, {args.out}_cross_cka.csv, {args.out}.png")


if __name__ == "__main__":
    main()