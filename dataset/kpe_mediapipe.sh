#!/bin/bash

for i in {0..255}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=kpe_$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH -G1
#SBATCH --cpus-per-task=1
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_kp_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_kp_%x.err

source ~/.bashrc
conda activate feature_extraction

# remember to download hand kpe model : wget https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
# remember to download face kpe model : wget https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models

python kpe_mediapipe.py --index "$i" \
                          --batch_size 1 \
                          --pose_path /hpchome/baptiste26/data/RUV/L1_signers/mediapipe/ \
                          --stats_path /hpchome/baptiste26/data/RUV/L1_signers/mediapipe/ \
                          --time_limit 257890 \
                          --files_list /hpchome/baptiste26/data/RUV/L1_signers/cropped/clips_bbox.list \
                          --problem_file_path /hpchome/baptiste26/data/RUV/L1_signers/mediapipe/problem.txt \
                          --face_model_path /hpchome/baptiste26/models_weights/mediapipe/face_landmarker.task \
                          --hand_model_path /hpchome/baptiste26/models_weights/mediapipe/hand_landmarker.task
EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


