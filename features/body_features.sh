#!/bin/bash

python ../wrist_list.py --input_dir /hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe \
    --output_dir /hpchome/baptiste26/data/RUV/news/SHuBERT/body_features \
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

MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/body_features

mkdir -p \$MAIN_PATH
FILE_LIST=\$MAIN_PATH/body_features.list

python body_features.py --index "$i" \
                        --batch_size 20 \
                        --files_list \$FILE_LIST \
                        --pose_features_path \$MAIN_PATH \
                        --time_limit 257890

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


