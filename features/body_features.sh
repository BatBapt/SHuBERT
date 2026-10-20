#!/bin/bash

for i in {0..4}; do
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

python body_features.py --index "$i" \
                          --batch_size 4 \
                          --files_list /hpchome/baptiste26/data/RUV/L1_signers/mediapipe/clips_bbox.list \
                          --pose_path /hpchome/baptiste26/data/RUV/L1_signers/mediapipe/ \
                          --pose_features_path /hpchome/baptiste26/data/RUV/L1_signers/body_features/ \
                          --time_limit 257890

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


