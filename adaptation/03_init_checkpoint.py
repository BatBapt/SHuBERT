"""
Step 4 of the SHuBERT adaptation pipeline.

Loads the pretrained SHuBERT checkpoint, re-initializes the 4 prediction heads
(their weights correspond to the ASL k-means clusters) and saves a new checkpoint
used as checkpoint.restore_file for the ISL adaptation.
All other weights (input projections, transformer, mask embedding) are kept.
"""
import argparse

import torch
import torch.nn as nn


def main(args):
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)

    print("Top-level keys:", list(ckpt.keys()))
    state = ckpt["model"]
    print(f"Model tensors: {len(state)}")

    head_keys = sorted(k for k in state if k.startswith("heads."))
    print("Head tensors:", {k: tuple(state[k].shape) for k in head_keys})
    assert len(head_keys) == 8, f"Expected 4 heads (weight + bias), found {head_keys}"

    for i in range(4):
        w, b = state[f"heads.{i}.weight"], state[f"heads.{i}.bias"]
        out_dim, in_dim = w.shape
        new_head = nn.Linear(in_dim, out_dim)  # default PyTorch init
        state[f"heads.{i}.weight"] = new_head.weight.data.to(w.dtype)
        state[f"heads.{i}.bias"] = new_head.bias.data.to(b.dtype)
    print("Re-initialized heads.0 to heads.3")

    # Optimizer state is reset at adaptation time (checkpoint.reset_optimizer=true)
    if "last_optimizer_state" in ckpt:
        ckpt["last_optimizer_state"] = None

    torch.save(ckpt, args.out_ckpt)
    print(f"Saved {args.out_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ckpt", required=True, help="checkpoint_836_400000.pt")
    parser.add_argument("--out_ckpt", required=True, help="e.g. checkpoint_isl_init.pt")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())