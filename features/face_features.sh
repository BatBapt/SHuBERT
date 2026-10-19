#!/bin/bash

for i in {0..1}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=facefeatures_$i
#SBATCH --partition=gpu-1xA100
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_face_features_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_face_features_%x.err

source ~/.bashrc
conda activate feature_extraction

python dinov2_features.py --index "$i" \
                          --batch_size 20 \
                          --files_list /hpchome/baptiste26/data/RUV/L1_signers/body_features/bbox_clips.list \
                          --output_folder /hpchome/baptiste26/data/RUV/L1_signers/face_dino/ \
                          --dino_path /hpchome/baptiste26/models_weights/shubert_weights/face_dinov2_checkpoint.pth \
                          --time_limit 257890

EOL
    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done


