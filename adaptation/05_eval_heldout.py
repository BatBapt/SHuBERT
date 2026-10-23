"""
Evaluation of SHuBERT base vs ISL-adapted checkpoints on the held-out videos.

Masked cluster prediction with a fixed mask (same seed for every model):
- own heads   : adapted checkpoints only, on train and held-out videos (overfitting check)
- fresh heads : 4 new linear heads trained on the frozen encoder output of the train videos,
                evaluated on held-out videos. Same protocol for base and adapted models
                -> fair base vs adapted comparison.

Run from <repo>/fairseq with PYTHONPATH=<repo>/fairseq
Outputs in out_dir: heldout_eval.csv, heldout_eval.png (mean of the 4 streams),
                    heldout_streams.csv, heldout_streams.png (per stream)
Use --from_csv heldout_eval.csv to redo the tables and plots without recomputing.
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairseq.data.data_utils import compute_mask_indices
from examples.shubert.models.shubert import SHubertModel, SHubertConfig

STREAMS = ["face", "left_hand", "right_hand", "body_posture"]  # same order as the model heads
MASK_PROB, MASK_LEN = 0.8, 3  # same as training (isl_adapt.yaml)
N_CLUSTERS = 256


def read_split(data_dir, split):
    videos = []
    with open(data_dir / f"{split}.tsv") as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) == 5:
                videos.append(items[:4])
    labels = {}
    for s in STREAMS:
        with open(data_dir / "labels" / f"{split}_{s}.km") as f:
            labels[s] = [np.array(l.split(), dtype=np.int64) for l in f]
        assert len(labels[s]) == len(videos), f"{split} {s}: label/video count mismatch"
    return videos, labels


def load_model(path, device):
    model = SHubertModel(SHubertConfig())
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{path}: missing keys {missing}")
    if unexpected:
        print(f"  unexpected keys ignored: {unexpected}")
    return model.eval().to(device)


def make_mask(T, seed):
    np.random.seed(seed)
    m = compute_mask_indices((1, T * 4), None, MASK_PROB, MASK_LEN, "static", 0.0,
                             min_masks=1, require_same_masks=True, mask_dropout=0.0)
    return m.reshape(T, 4)  # True = masked, columns in STREAMS order


@torch.no_grad()
def extract(model, videos, labels, chunk, device, seed):
    feats, masks, labs = [], [], {s: [] for s in STREAMS}
    for i, paths in enumerate(videos):
        arrs = [np.load(p).astype(np.float32) for p in paths]
        T = min(a.shape[0] for a in arrs)
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            if end - start < 10:
                continue
            m = make_mask(end - start, seed + 100000 * i + start)
            src = {s: torch.from_numpy(a[start:end]).to(device) for s, a in zip(STREAMS, arrs)}
            for s in STREAMS:
                src[f"label_{s}"] = torch.zeros(end - start, 1, device=device)
            res = model.forward([src], padding_mask=None, mask=True, features_only=True,
                                mask_indices=torch.from_numpy(m.reshape(1, -1)).to(device))
            feats.append(res["x"][0].float().cpu())
            masks.append(torch.from_numpy(m))
            for s in STREAMS:
                labs[s].append(torch.from_numpy(labels[s][i][start:end]))
    return torch.cat(feats), torch.cat(masks), {s: torch.cat(v) for s, v in labs.items()}


@torch.no_grad()
def evaluate(heads, X, M, Y, device):
    out = {}
    for c, s in enumerate(STREAMS):
        idx = M[:, c]
        logits = heads[c](X[idx].to(device))
        y = Y[s][idx].to(device)
        out[s] = (F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item())
    out["mean"] = tuple(np.mean([out[s][k] for s in STREAMS]) for k in range(2))
    return out


def train_fresh_heads(X, M, Y, epochs, device, seed):
    torch.manual_seed(seed)
    heads = []
    for c, s in enumerate(STREAMS):
        x, y = X[M[:, c]].to(device), Y[s][M[:, c]].to(device)
        head = nn.Linear(x.shape[1], N_CLUSTERS).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            perm = torch.randperm(len(x), device=device)
            for b in range(0, len(x), 4096):
                j = perm[b:b + 4096]
                loss = F.cross_entropy(head(x[j]), y[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
        heads.append(head.eval())
    return heads


def list_models(base_ckpt, ckpt_dir):
    models = [("base", 0, base_ckpt)]
    for p in Path(ckpt_dir).glob("checkpoint_*_*.pt"):
        m = re.match(r"checkpoint_(\d+)_(\d+)\.pt", p.name)
        if m:
            models.append((f"adapted_{m.group(2)}", int(m.group(2)), str(p)))
    return sorted(models, key=lambda x: x[1])


def run_eval(args, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    tr_videos, tr_labels = read_split(data_dir, "train")
    ho_videos, ho_labels = read_split(data_dir, "heldout")
    print(f"{len(tr_videos)} train videos, {len(ho_videos)} held-out videos, device {device}")

    rows = []
    for name, update, path in list_models(args.base_ckpt, args.ckpt_dir):
        print(f"\n== {name} ({path})")
        model = load_model(path, device)
        Xtr, Mtr, Ytr = extract(model, tr_videos, tr_labels, args.chunk, device, args.seed)
        Xho, Mho, Yho = extract(model, ho_videos, ho_labels, args.chunk, device, args.seed + 1)

        results = []
        if name != "base":  # base heads predict ASL clusters, not comparable
            results.append(("own", "train", evaluate(model.heads, Xtr, Mtr, Ytr, device)))
            results.append(("own", "heldout", evaluate(model.heads, Xho, Mho, Yho, device)))
        fresh = train_fresh_heads(Xtr, Mtr, Ytr, args.fresh_epochs, device, args.seed)
        results.append(("fresh", "heldout", evaluate(fresh, Xho, Mho, Yho, device)))

        for head_type, split, res in results:
            for stream, (loss, acc) in res.items():
                rows.append({"model": name, "update": update, "heads": head_type, "split": split,
                             "stream": stream, "loss": round(loss, 4), "acc": round(acc, 4)})
            print(f"  {head_type:5s} {split:7s} mean loss {res['mean'][0]:.3f}  mean acc {res['mean'][1]:.3f}")

        del model, Xtr, Xho
        torch.cuda.empty_cache()

    with open(out_dir / "heldout_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["update"], r["loss"], r["acc"] = int(r["update"]), float(r["loss"]), float(r["acc"])
            rows.append(r)
    return rows


def series(rows, heads, split, key, stream="mean"):
    pts = sorted((r["update"], r[key]) for r in rows
                 if r["heads"] == heads and r["split"] == split and r["stream"] == stream and r["model"] != "base")
    return [p[0] for p in pts], [p[1] for p in pts]


def base_value(rows, key, stream="mean"):
    return [r[key] for r in rows if r["model"] == "base" and r["heads"] == "fresh" and r["stream"] == stream][0]


def report_streams(rows, out_dir):
    """Per-stream table (fresh heads, held-out): base vs best and last adapted checkpoint."""
    fresh = [r for r in rows if r["heads"] == "fresh" and r["split"] == "heldout" and r["model"] != "base"]
    best = max((r for r in fresh if r["stream"] == "mean"), key=lambda r: r["acc"])
    last = max((r for r in fresh if r["stream"] == "mean"), key=lambda r: r["update"])

    lines = []
    header = f"{'stream':14s} {'base acc':>9s} {'best acc':>9s} {'last acc':>9s} {'d_last':>8s} {'base loss':>10s} {'last loss':>10s} {'d_loss':>8s}"
    print(f"\nPer stream, fresh heads, held-out (best = {best['model']}, last = {last['model']})")
    print(header)
    for s in STREAMS + ["mean"]:
        get = lambda name, key: [r[key] for r in fresh if r["model"] == name and r["stream"] == s][0]
        b_acc, b_loss = base_value(rows, "acc", s), base_value(rows, "loss", s)
        bst, l_acc, l_loss = get(best["model"], "acc"), get(last["model"], "acc"), get(last["model"], "loss")
        print(f"{s:14s} {b_acc:9.3f} {bst:9.3f} {l_acc:9.3f} {l_acc - b_acc:+8.3f} {b_loss:10.3f} {l_loss:10.3f} {l_loss - b_loss:+8.3f}")
        lines.append({"stream": s, "base_acc": b_acc, "best_acc": bst, "last_acc": l_acc, "delta_acc": round(l_acc - b_acc, 4),
                      "base_loss": b_loss, "last_loss": l_loss, "delta_loss": round(l_loss - b_loss, 4)})

    with open(out_dir / "heldout_streams.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0].keys()))
        w.writeheader()
        w.writerows(lines)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, s in zip(axes.flat, STREAMS):
        ax.plot(*series(rows, "fresh", "heldout", "acc", s), "o-", label="adapted (fresh heads)")
        ax.plot(*series(rows, "own", "heldout", "acc", s), "s--", alpha=0.6, label="adapted (own heads)")
        ax.axhline(base_value(rows, "acc", s), color="gray", ls="--", label="base (fresh heads)")
        ax.set_title(f"{s}: held-out accuracy")
        ax.set_xlabel("updates")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_streams.png", dpi=120)
    print(f"Saved {out_dir / 'heldout_streams.csv'} and {out_dir / 'heldout_streams.png'}")


def plot_summary(rows, out_dir):
    """Mean over the 4 streams."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(*series(rows, "own", "train", "loss"), "o-", label="own heads - train")
    ax.plot(*series(rows, "own", "heldout", "loss"), "o-", label="own heads - held-out")
    ax.set_title("Adapted checkpoints: masked prediction loss")
    ax.set_xlabel("updates")
    ax.set_ylabel("cross-entropy (nats)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(*series(rows, "fresh", "heldout", "acc"), "o-", label="adapted (fresh heads)")
    ax.axhline(base_value(rows, "acc"), color="gray", ls="--", label="base (fresh heads)")
    ax.axhline(1 / N_CLUSTERS, color="red", ls=":", label="chance")
    ax.set_title("Held-out accuracy, fresh heads: base vs adapted")
    ax.set_xlabel("updates")
    ax.set_ylabel("accuracy (mean of 4 streams)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_eval.png", dpi=120)
    print(f"\nSaved {out_dir / 'heldout_eval.png'}")


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        rows = load_rows(args.from_csv)
        print(f"Loaded {len(rows)} rows from {args.from_csv} (no recomputation)")
    else:
        if not (args.data_dir and args.ckpt_dir and args.base_ckpt):
            raise SystemExit("--data_dir, --ckpt_dir and --base_ckpt are required without --from_csv")
        rows = run_eval(args, out_dir)
    plot_summary(rows, out_dir)
    report_streams(rows, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_csv", default=None, help="existing heldout_eval.csv: only redo tables and plots")
    parser.add_argument("--data_dir", help="dir with train.tsv, heldout.tsv, labels/")
    parser.add_argument("--ckpt_dir", help="run_1/ckpt")
    parser.add_argument("--base_ckpt", help="checkpoint_836_400000.pt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--chunk", type=int, default=1500, help="frames per forward pass (= max_sample_size)")
    parser.add_argument("--fresh_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())