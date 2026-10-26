#!/bin/bash

MODE=$1

case "$MODE" in
    news)
        POSE_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/cropped_hands
        MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/hands_dino
        FILE_LIST=/hpchome/baptiste26/data/RUV/news/SHuBERT/hands_dino/hands_features.list
        ;;
    L1)
        POSE_PATH=/hpchome/baptiste26/data/RUV/L1_signers/cropped_hands
        MAIN_PATH=/hpchome/baptiste26/data/RUV/L1_signers/hands_dino
        FILE_LIST=/hpchome/baptiste26/data/RUV/L1_signers/hands_dino/hands_features.list
        ;;
    *)
        echo "Usage: $0 {news|L1}"
        exit 1
        ;;
esac

python ../write_list.py --input_dir $POSE_PATH \
    --output_path $MAIN_PATH \
    --output_file_name hands_features \
    --ext .mp4 || exit 1

for i in {0..1}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=hands_features_$i
#SBATCH --partition=gpu-1xA100
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_hands_features_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_hands_features_%x.err
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate dino

mkdir -p \$MAIN_PATH

python dinov2_features.py --index "$i" \
                          --batch_size 5000 \
                          --files_list $\FILE_LIST \
                          --output_folder $\MAIN_PATH/ \
                          --dino_path /hpchome/baptiste26/models_weights/shubert_weights/hands_dinov2_checkpoint.pth \
                          --time_limit 257890

EOL

    sbatch temp_slurm_script_$i.sh
    rm temp_slurm_script_$i.sh
done


