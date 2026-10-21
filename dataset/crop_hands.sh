#!/bin/bash

for i in {0..6}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=crop_hands_$i
#SBATCH --partition=128cpu_256mem
#SBATCH --cpus-per-task=16
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_crops_hands_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_crops_hands_%x.err

source ~/.bashrc
conda activate feature_extraction

MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/cropped_hands

mkdir -p \$MAIN_PATH
FILE_LIST=/hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe/cropped.list

PROBLEM_FILE_PATH=\$MAIN_PATH/problem.txt
touch \$PROBLEM_FILE_PATH

python crop_hands.py --index "$i" \
                          --batch_size 20 \
                          --files_list \$FILE_LIST \
                          --problem_file_path \$PROBLEM_FILE_PATH \
                          --pose_path /hpchome/baptiste26/data/RUV/L1_signers/mediapipe/ \
                          --hand_path \$MAIN_PATH \
                          --time_limit 257890

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


