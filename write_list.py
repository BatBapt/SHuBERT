import gzip
import pickle
import os

input_dir = "/hpchome/baptiste26/data/RUV/L1_signers/original"
input_files_list = sorted(
    os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".mxf")
)

output_path = "/hpchome/baptiste26/data/RUV/L1_signers/cropped"
os.makedirs(output_path, exist_ok=True)

output_file = os.path.join(output_path, "clips_bbox.list")
with gzip.GzipFile(output_file, 'wb') as file_b:
    file_b.write(pickle.dumps(input_files_list, protocol=0))