"""
Steps 2-3 of the SHuBERT adaptation pipeline.

- Splits the videos of videos_12fps.tsv into adaptation (train) / held-out
- Fits one k-means (256 clusters) per stream on the train videos only
- Writes the fairseq manifest and label files expected by train_shubert.sh:
    out_dir/train.tsv, out_dir/heldout.tsv
    out_dir/labels/{train,heldout}_{stream}.km   (one line per video, one label per frame)
    out_dir/kmeans/{stream}.joblib               (fitted k-means models)
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

STREAMS = ["face", "left_hand", "right_hand", "body_posture"]  # same order as the manifest columns


def read_tsv(path):
    videos = []
    with open(path) as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) != 5:
                continue
            vid = Path(items[0]).stem
            videos.append({"id": vid, "paths": dict(zip(STREAMS, items[:4])), "n": int(items[4]), "line": line.strip()})
    return sorted(videos, key=lambda v: v["id"])


def fit_kmeans(feats, n_clusters, seed):
    # Same settings as fairseq/examples/hubert/simple_kmeans/learn_kmeans.py
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=100,
        batch_size=10000,
        tol=0.0,
        max_no_improvement=100,
        n_init=3,
        reassignment_ratio=0.0,
        random_state=seed,
        verbose=1,
    )
    km.fit(feats)
    return km


def write_labels(km, videos, stream, out_path):
    with open(out_path, "w") as f:
        for v in videos:
            x = np.load(v["paths"][stream]).astype(np.float32)
            assert x.shape[0] == v["n"], f"{v['id']} {stream}: {x.shape[0]} frames vs {v['n']} in tsv"
            labels = km.predict(x)
            f.write(" ".join(map(str, labels)) + "\n")


def main(args):
    out_dir = Path(args.out_dir)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "kmeans").mkdir(parents=True, exist_ok=True)

    videos = read_tsv(args.tsv)
    train, heldout = videos[: -args.n_heldout], videos[-args.n_heldout:]
    print(f"{len(train)} train videos, {len(heldout)} held-out videos")
    print("Held-out:", [v["id"] for v in heldout])

    for name, subset in [("train", train), ("heldout", heldout)]:
        with open(out_dir / f"{name}.tsv", "w") as f:
            f.write("\n".join(v["line"] for v in subset) + "\n")

    for stream in STREAMS:
        feats = np.concatenate([np.load(v["paths"][stream]).astype(np.float32) for v in train])
        print(f"[{stream}] fitting k-means on {feats.shape[0]} frames, dim {feats.shape[1]}")
        km = fit_kmeans(feats, args.n_clusters, args.seed)
        joblib.dump(km, out_dir / "kmeans" / f"{stream}.joblib")

        used = len(np.unique(km.predict(feats)))
        print(f"[{stream}] clusters used on train: {used}/{args.n_clusters}")

        write_labels(km, train, stream, out_dir / "labels" / f"train_{stream}.km")
        write_labels(km, heldout, stream, out_dir / "labels" / f"heldout_{stream}.km")

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="videos_12fps.tsv from subsample_features.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_heldout", type=int, default=4)
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())