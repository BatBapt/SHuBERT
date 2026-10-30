import torch
import numpy as np
import csv
import os
from tqdm import tqdm
import torch
import argparse
from examples.shubert.models.shubert import SHubertModel, SHubertConfig

# Function to load the model
def load_model(checkpoint_path):
    cfg = SHubertConfig() 
    
    # Initialize the model
    model = SHubertModel(cfg)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # If the checkpoint is saved with a 'model' key
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load the state dictionary into the model
    res = model.load_state_dict(state_dict, strict=False)
    print(f"[load] missing: {len(res.missing_keys)}  unexpected: {len(res.unexpected_keys)}")
    print("[load] first missing:", res.missing_keys[:5])
    print("[load] first unexpected:", res.unexpected_keys[:5])
    
    model.eval()
    model.cuda()  # Move to GPU if available
    return model


# Function to process a single sample
def process_sample(model, face_path, left_hand_path, right_hand_path, body_posture_path, stride=2):
    # Load numpy arrays
    face_np = np.load(face_path)
    left_hand_np = np.load(left_hand_path)
    right_hand_np = np.load(right_hand_path)
    body_posture_np = np.load(body_posture_path)

    # Same length on all streams, then keep one frame every `stride`
    lengths = [face_np.shape[0], left_hand_np.shape[0], right_hand_np.shape[0], body_posture_np.shape[0]]
    if len(set(lengths)) > 1:
        print(f"Warning: stream lengths differ {lengths} for {face_path}, truncating to {min(lengths)}")
    n = min(lengths)
    face_np = face_np[:n:stride]
    left_hand_np = left_hand_np[:n:stride]
    right_hand_np = right_hand_np[:n:stride]
    body_posture_np = body_posture_np[:n:stride]

    face = torch.from_numpy(face_np).float().cuda()
    left_hand = torch.from_numpy(left_hand_np).float().cuda()
    right_hand = torch.from_numpy(right_hand_np).float().cuda()
    body_posture = torch.from_numpy(body_posture_np).float().cuda()

    length = face.shape[0]
    # Prepare input
    source = [{
        "face": face,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "body_posture": body_posture,
        # Add dummy labels to match the expected input format
        "label_face": torch.zeros((length, 1)).cuda(),
        "label_left_hand": torch.zeros((length, 1)).cuda(),
        "label_right_hand": torch.zeros((length, 1)).cuda(),
        "label_body_posture": torch.zeros((length, 1)).cuda()
    }]

    # Extract features
    with torch.no_grad():
        result = model.extract_features(source, padding_mask=None, kmeans_labels=None, mask=False)

    # Extract layer outputs
    layer_outputs = []
    for layer in result['layer_results']:
        # layer_output has shape [T, B, D], B = 1
        layer_output = layer[-1].squeeze(1)  # Shape: [T, D]
        layer_outputs.append(layer_output.cpu().numpy())

    # Stack the outputs from all layers: [L, T, D]
    features = np.stack(layer_outputs, axis=0)
    return features

# Main script
def main(csv_list, checkpoint_path, output_dir, index, stride=2):
    model = load_model(checkpoint_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
        
    for row in csv_list:

        cues_list = row[0].split('\t')
        face_path, left_hand_path, right_hand_path, body_posture_path = cues_list[0], cues_list[1], cues_list[2], cues_list[3]

        output_filename = f"{os.path.basename(face_path).rsplit('.',1)[0].rsplit('_',1)[0]}.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        # check if the output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_path} as it already exists")
            continue
        
        # Process the sample
        features = process_sample(model, face_path, left_hand_path, right_hand_path, body_posture_path, stride)

        # np.save(output_path, features)
        np.save(output_path, np.asarray(features, dtype=np.float16))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True,
                        help='index of the sub_list to work with')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='path to the CSV file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='directory to save output files')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size for processing')
    parser.add_argument('--stride', type=int, default=2,
                        help='stride for downsampling the input sequences')
    
    args = parser.parse_args()
    index = args.index
    csv_path = args.csv_path
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    batch_size = int(args.batch_size)
    stride = int(args.stride)

    # make output dir
    os.makedirs(output_dir, exist_ok=True)
    
    fixed_list = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fixed_list.append(row)
    
    
    video_batches = [fixed_list[i:i + batch_size] for i in range(0, len(fixed_list), batch_size)]
    
    csv_list = video_batches[index]
    main(csv_list, checkpoint_path, output_dir, index, stride)
