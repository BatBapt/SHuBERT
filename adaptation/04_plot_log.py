"""
Plots the training curves of a fairseq (SHuBERT) run from log.txt or the SLURM .out file.
Duplicated lines are removed (one entry per update / per epoch).

Outputs in out_dir: train_curves.png, train_inner.csv, train_epochs.csv
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_float(d):
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def parse_log(path):
    inner, epochs = {}, {}
    with open(path, errors="ignore") as f:
        for line in f:
            start = line.find("{")
            if start < 0:
                continue
            try:
                d = to_float(json.loads(line[start:]))
            except json.JSONDecodeError:
                continue
            if "train_num_updates" in d:      # end-of-epoch stats
                epochs[int(d["epoch"])] = d
            elif "num_updates" in d and "loss" in d:  # stats every log_interval updates
                inner[int(d["num_updates"])] = d
    return [inner[k] for k in sorted(inner)], [epochs[k] for k in sorted(epochs)]


def write_csv(rows, path):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inner, epochs = parse_log(args.log)
    print(f"{len(inner)} unique inner entries, {len(epochs)} unique epoch entries")
    if not inner:
        print("No training entries found")
        return

    write_csv(inner, out_dir / "train_inner.csv")
    write_csv(epochs, out_dir / "train_epochs.csv")

    upd = [r["num_updates"] for r in inner]
    loss = [r["loss"] for r in inner]
    i_min = min(range(len(loss)), key=lambda i: loss[i])

    print("\nSummary")
    print(f"  updates          : {int(upd[0])} -> {int(upd[-1])}")
    print(f"  epochs           : {len(epochs)}")
    print(f"  loss first / last: {loss[0]:.3f} / {loss[-1]:.3f}")
    print(f"  loss min         : {loss[i_min]:.3f} at update {int(upd[i_min])}")
    if epochs and "train_wall" in epochs[-1]:
        print(f"  wall time        : {epochs[-1]['train_wall'] / 60:.1f} min")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("loss", "Train loss (masked cluster prediction)"),
              ("lr", "Learning rate"),
              ("gnorm", "Gradient norm"),
              ("wps", "Throughput (tokens / s)")]
    for ax, (key, title) in zip(axes.flat, panels):
        x = [r["num_updates"] for r in inner if key in r]
        y = [r[key] for r in inner if key in r]
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel("updates")
        ax.grid(alpha=0.3)
    fig.suptitle(Path(args.log).parent.name)
    fig.tight_layout()
    fig.savefig(out_dir / "train_curves.png", dpi=120)
    print(f"\nSaved {out_dir / 'train_curves.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="log.txt or SLURM .out file")
    parser.add_argument("--out_dir", required=True)
    main(parser.parse_args())