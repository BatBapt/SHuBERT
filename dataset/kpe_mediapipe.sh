#!/bin/bash

MODE=$1

case "$MODE" in
    news)
        INPUT_DIR=/hpchome/baptiste26/data/RUV/news/SHuBERT/cropped
        MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe
        N_JOBS=60
        BATCH_SIZE=2
        ;;
    L1)
        INPUT_DIR=/hpchome/baptiste26/data/RUV/L1_signers/cropped
        MAIN_PATH=/hpchome/baptiste26/data/RUV/L1_signers/mediapipe
        N_JOBS=3
        BATCH_SIZE=20
        ;;
    *)
        echo "Usage: $0 {news|L1}"
        exit 1
        ;;
esac

python ../write_list.py --input_dir $INPUT_DIR \
    --output_path $MAIN_PATH \
    --output_file_name cropped \
    --ext mp4 || exit 1

for i in $(seq 0 $((N_JOBS - 1))); do
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=kpe_${MODE}_$i
#SBATCH --partition=48cpu_192mem
#SBATCH --cpus-per-task=$((2 * BATCH_SIZE))
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_kp_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_kp_%x.err

source ~/.bashrc
conda activate feature_extraction

mkdir -p $MAIN_PATH
touch $MAIN_PATH/problem.txt

for k in \$(seq 0 $((BATCH_SIZE - 1))); do
  python kpe_mediapipe.py --index \$(($BATCH_SIZE*$i + k)) \\
                          --batch_size 1 \\
                          --pose_path $MAIN_PATH \\
                          --stats_path $MAIN_PATH \\
                          --files_list $MAIN_PATH/cropped.list \\
                          --problem_file_path $MAIN_PATH/problem.txt \\
                          --face_model_path /hpchome/baptiste26/models_weights/mediapipe/face_landmarker.task \\
                          --hand_model_path /hpchome/baptiste26/models_weights/mediapipe/hand_landmarker.task \\
                          --time_limit 257890 &
done
wait
EOL
    sbatch temp_slurm_script_$i.sh
    rm temp_slurm_script_$i.sh
done