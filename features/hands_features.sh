#!/bin/bash

python ../wrist_list.py --input_dir /hpchome/baptiste26/data/RUV/news/SHuBERT/cropped_hands \
    --output_dir /hpchome/baptiste26/data/RUV/news/SHuBERT/hands_dino \
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
#SBATCH --time=45:00

source ~/.bashrc
conda activate dino

MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/hands_dino

FILE_LIST=$MAIN_PATH/hands_features.list

python dinov2_features.py --index "$i" \
                          --batch_size 5000 \
                          --files_list $FILE_LIST \
                          --output_folder $MAIN_PATH/ \
                          --dino_path /hpchome/baptiste26/models_weights/shubert_weights/hands_dinov2_checkpoint.pth \
                          --time_limit 257890

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


