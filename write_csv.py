import os

data_path = "/hpchome/baptiste26/data/RUV/L1_signers"
out = f"{data_path}/shubert_inputs.csv"

with open(out, "w") as f:
    for name in sorted(os.listdir(f"{data_path}/face_dino")):
        vid = name.rsplit("_", 1)[0]  # 027721-0001-01
        paths = [
            f"{data_path}/face_dino/{vid}_face.npy",
            f"{data_path}/hands_dino/{vid}_hand1.npy",
            f"{data_path}/hands_dino/{vid}_hand2.npy",
            f"{data_path}/body_features/{vid}_pose.npy",
        ]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print("Missing:", missing)
            continue
        f.write("\t".join(paths) + "\n")