"""
Step 1 of the SHuBERT adaptation pipeline.

Keeps one frame every `stride` (25 fps -> 12.5 fps) in the 4 input streams
(face, left hand, right hand, body) and writes the result to out_dir.
Also writes videos_12fps.tsv: one line per video,
face \t left_hand \t right_hand \t body_posture \t n_frames
"""
import argparse
import os
from pathlib import Path

import numpy as np

# File name suffixes: adjust if your files are named differently.
# hand1 = left hand, hand2 = right hand (see crop_hands.py)
FACE_SUFFIX = "_face.npy"
LHAND_SUFFIX = "_hand1.npy"
RHAND_SUFFIX = "_hand2.npy"
BODY_SUFFIX = "_pose.npy"


def main(args):
    out_dir = Path(args.out_dir)
    streams = {
        "face": (Path(args.face_dir), FACE_SUFFIX),
        "left_hand": (Path(args.hands_dir), LHAND_SUFFIX),
        "right_hand": (Path(args.hands_dir), RHAND_SUFFIX),
        "body_posture": (Path(args.body_dir), BODY_SUFFIX),
    }
    for name in streams:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    video_ids = sorted(p.name[: -len(FACE_SUFFIX)] for p in Path(args.face_dir).glob(f"*{FACE_SUFFIX}"))
    print(f"Found {len(video_ids)} videos in {args.face_dir}")

    lines, errors = [], []
    for vid in video_ids:
        paths = {name: d / f"{vid}{suf}" for name, (d, suf) in streams.items()}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            errors.append(f"{vid}: missing {missing}")
            continue

        feats = {name: np.load(p) for name, p in paths.items()}
        lengths = {name: f.shape[0] for name, f in feats.items()}
        n = min(lengths.values())
        if len(set(lengths.values())) > 1:
            print(f"Warning: {vid} stream lengths differ {lengths}, truncating to {n}")

        out_paths = []
        for name, f in feats.items():
            out_path = out_dir / name / f"{vid}.npy"
            np.save(out_path, f[:n:args.stride].astype(np.float32))
            out_paths.append(str(out_path.resolve()))

        n_out = len(range(0, n, args.stride))
        lines.append("\t".join(out_paths + [str(n_out)]))
        print(f"{vid}: {n} -> {n_out} frames")

    with open(out_dir / "videos_12fps.tsv", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Done: {len(lines)} videos written, {len(errors)} errors")
    for e in errors:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_dir", required=True)
    parser.add_argument("--hands_dir", required=True)
    parser.add_argument("--body_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stride", type=int, default=2)
    main(parser.parse_args())