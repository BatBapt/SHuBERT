#!/bin/bash

python /hpchome/baptiste26/my_code/SHuBERT/wrist_list.py --input_dir /hpchome/baptiste26/data/RUV/news/SHuBERT/cropped \
    --output_dir /hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe \
    --output_file_name cropped \
    --ext mp4 || exit 1

for i in {0..6}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=kpe_$i
#SBATCH --partition=128cpu_256mem
#SBATCH --cpus-per-task=16
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_kp_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_kp_%x.err

source ~/.bashrc
conda activate feature_extraction

MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe

mkdir -p \$MAIN_PATH
FILE_LIST=\$MAIN_PATH/cropped.list

PROBLEM_FILE_PATH=\$MAIN_PATH/problem.txt
touch \$PROBLEM_FILE_PATH

python kpe_mediapipe.py --index "$i" \
                        --batch_size 20 \
                        --pose_path \$MAIN_PATH \
                        --stats_path \$MAIN_PATH \
                        --files_list \$MAIN_PATH/mediapipe.list \
                        --problem_file_path \$PROBLEM_FILE_PATH \
                        --face_model_path /hpchome/baptiste26/models_weights/mediapipe/face_landmarker.task \
                        --hand_model_path /hpchome/baptiste26/models_weights/mediapipe/hand_landmarker.task \
                        --time_limit 257890
EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


