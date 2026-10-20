import gzip
import pickle
import os
import argparse

if __name__ == "__main__":
    # input_dir = "/hpchome/baptiste26/data/RUV/L1_signers/original"
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing .mxf files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory where the clips_bbox.list will be saved.")
    parser.add_argument("--output_file_name", type=str, required=True, help="Name of the output file (without extension).")
    parser.add_argument("--ext", type=str, required=True, default=".mxf", help="File extension for processing.")
    args = parser.parse_args()
    
    input_files_list = sorted(
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(args.ext)
    )
    
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, f"{args.output_file_name}.list")
    with gzip.GzipFile(output_file, 'wb') as file_b:
        file_b.write(pickle.dumps(input_files_list, protocol=0))