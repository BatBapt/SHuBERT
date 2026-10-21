#!/bin/bash

python ../wrist_list.py --input_dir /hpchome/baptiste26/data/RUV/news/ruv_original \
    --output_dir /hpchome/baptiste26/data/RUV/news/SHuBERT/cropped \
    --output_file_name clips_bbox \
    --ext mp4 || exit 1

for i in {0..6}; do
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=crop_$i
#SBATCH --partition=128cpu_256mem
#SBATCH --cpus-per-task=16
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_crops_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_crops_%x.err

source ~/.bashrc
conda activate feature_extraction

MAIN_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/cropped

mkdir -p \$MAIN_PATH
FILE_LIST=\$MAIN_PATH/clips_bbox.list

PROBLEM_FILE_PATH=\$MAIN_PATH/problem.txt
touch \$PROBLEM_FILE_PATH

python clips_bbox.py --index "$i" \
    --batch_size 20 \
    --files_list \$FILE_LIST \
    --output_clips_directory \$MAIN_PATH \
    --problem_file_path \$PROBLEM_FILE_PATH \
    --yolo_model_path /hpchome/baptiste26/models_weights/yolov8n.pt
EOL

    sbatch temp_slurm_script_$i.sh
    rm temp_slurm_script_$i.sh
done