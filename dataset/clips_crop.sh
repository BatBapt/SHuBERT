#!/bin/bash

for i in {0..5}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=crop_$i
#SBATCH --partition=128cpu_256mem
#SBATCH --cpus-per-task=16
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_%x.err

source ~/.bashrc
conda activate feature_extraction

python clips_bbox.py --index "$i" \
    --batch_size 50 \
    --files_list /hpchome/baptiste26/data/RUV/L1_signers/cropped/clips_bbox.list \
    --output_clips_directory /hpchome/baptiste26/data/RUV/L1_signers/cropped \
    --problem_file_path /hpchome/baptiste26/data/RUV/L1_signers/cropped/problem.txt \
    --yolo_model_path /hpchome/baptiste26/models_weights/yolov8n.pt

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


