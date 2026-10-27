#!/bin/bash

MODE=$1

case "$MODE" in
    news)
        POSE_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe
        MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/cropped_face
        FILE_LIST=/hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe/cropped.list
        N_JOBS=60
        BATCH_SIZE=2
        ;;
    L1)
        POSE_PATH=/hpchome/baptiste26/data/RUV/L1_signers/mediapipe
        MAIN_PATH=/hpchome/baptiste26/data/RUV/L1_signers/cropped_face
        FILE_LIST=/hpchome/baptiste26/data/RUV/L1_signers/mediapipe/cropped.list
        N_JOBS=3
        BATCH_SIZE=20
        ;;
    *)
        echo "Usage: $0 {news|L1}"
        exit 1
        ;;
esac

for i in $(seq 0 $((N_JOBS - 1))); do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=crop_face_$i
#SBATCH --partition=48cpu_192mem
#SBATCH --cpus-per-task=$((2 * BATCH_SIZE))
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_crops_face_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_crops_face_%x.err

source ~/.bashrc
conda activate feature_extraction

mkdir -p $MAIN_PATH
touch $MAIN_PATH/problem.txt

for k in \$(seq 0 $((BATCH_SIZE - 1))); do
  python crop_face.py --index \$(($BATCH_SIZE*$i + k)) \\
                      --batch_size 1 \\
                      --files_list $FILE_LIST \\
                      --problem_file_path $MAIN_PATH/problem.txt \\
                      --pose_path $POSE_PATH \\
                      --face_path $MAIN_PATH \\
                      --time_limit 257890 &
done
wait
EOL
    sbatch temp_slurm_script_$i.sh
    rm temp_slurm_script_$i.sh
done