#!/bin/bash

for i in {0..4}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=crop_face_$i
#SBATCH --partition=128cpu_256mem
#SBATCH --cpus-per-task=16
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_crops_face_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_crops_face_%x.err

source ~/.bashrc
conda activate feature_extraction

MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/cropped_face

mkdir -p \$MAIN_PATH
FILE_LIST=/hpchome/baptiste26/data/RUV/news/SHuBERT/mediapipe/cropped.list

PROBLEM_FILE_PATH=\$MAIN_PATH/problem.txt
touch \$PROBLEM_FILE_PATH

python crop_face.py --index "$i" \
                    --batch_size 4 \
                    --files_list /hpchome/baptiste26/data/RUV/L1_signers/mediapipe/clips_bbox.list \
                    --problem_file_path /hpchome/baptiste26/data/RUV/L1_signers/cropped_face/problem.txt \
                    --pose_path /hpchome/baptiste26/data/RUV/L1_signers/mediapipe/ \
                    --face_path /hpchome/baptiste26/data/RUV/L1_signers/cropped_face/ \
                    --time_limit 257890

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


