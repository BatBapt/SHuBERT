#!/bin/bash

MODE=$1

case "$MODE" in
    news)
        POSE_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe
        MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/body_features
        FILE_LIST=/hpchome/baptiste26/data/RUV/news/SHuBERT/body_features/body_features.list
        N_JOBS=7
        ;;
    L1)
        POSE_PATH=/hpchome/baptiste26/data/RUV/L1_signers/mediapipe
        MAIN_PATH=/hpchome/baptiste26/data/RUV/L1_signers/body_features
        FILE_LIST=/hpchome/baptiste26/data/RUV/L1_signers/body_features/body_features.list
        N_JOBS=3
        ;;
    *)
        echo "Usage: $0 {news|L1}"
        exit 1
        ;;
esac

python ../write_list.py --input_dir $POSE_PATH \
    --output_path $MAIN_PATH \
    --output_file_name body_features \
    --ext _pose.json || exit 1

for i in {0..6}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=body_features_$i
#SBATCH --partition=128cpu_256mem
#SBATCH --cpus-per-task=16
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_body_features_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_body_features_%x.err

source ~/.bashrc
conda activate feature_extraction

mkdir -p $MAIN_PATH

python body_features.py --index "$i" \
                        --batch_size 20 \
                        --files_list $FILE_LIST \
                        --pose_features_path $MAIN_PATH \
                        --time_limit 257890

EOL

    sbatch temp_slurm_script_$i.sh
    rm temp_slurm_script_$i.sh
done


