#!/bin/bash

MODE=$1

case "$MODE" in
    news)
        DATA_PATH=/hpchome/baptiste26/data/RUV/news/SHuBERT/
        ;;
    L1)
        DATA_PATH=/hpchome/baptiste26/data/RUV/L1_signers/
        ;;
    *)
        echo "Usage: $0 {news|L1}"
        exit 1
        ;;
esac

python ../write_csv.py --data_path $DATA_PATH || exit 1

for i in {0..0}; do
    cat >tempGr_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=shubert_inference
#SBATCH --partition=gpu-1xA100
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_shubert_inference_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/slurm_shubert_inference_%x.err
#SBATCH --time=03:00:00

source ~/.bashrc
conda activate shubert_train_env

# Ensure fairseq_root is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/hpchome/baptiste26/my_code/SHuBERT/fairseq

python shubert_inference.py \\
    --index "$i" \\
    --csv_path $DATA_PATH/shubert_inputs.csv \\
    --checkpoint_path /hpchome/baptiste26/models_weights/shubert_weights/checkpoint_836_400000.pt \\
    --output_dir $DATA_PATH/shubert_outputs \\
    --batch_size 1000

EOL
    sbatch tempGr_slurm_script_$i.sh
    rm tempGr_slurm_script_$i.sh
done
