#!/bin/bash
#SBATCH --job-name=adapting_shubert
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00
#SBATCH --output=/hpchome/baptiste26/my_code/SHuBERT/logs/adapting_shubert_%x.out
#SBATCH --error=/hpchome/baptiste26/my_code/SHuBERT/logs/adapting_shubert_%x.err

source ~/.bashrc
conda activate shubert_train_env

ulimit -n 65535
export NCCL_TIMEOUT=1200
export WORLD_SIZE=${SLURM_NTASKS:-1}

REPO=/hpchome/baptiste26/my_code/SHuBERT
CONFIG_DIR=$REPO/fairseq/examples/shubert/config
CONFIG_NAME=isl_adapt
LOG_DIR=$REPO/run_2

mkdir -p $LOG_DIR
cd $REPO/fairseq

PYTHONPATH=$PYTHONPATH:$REPO/fairseq:$REPO/fairseq/examples \
    srun fairseq-hydra-train \
    --config-dir $CONFIG_DIR \
    --config-name $CONFIG_NAME \
    common.user_dir=$REPO/fairseq/examples/shubert \
    checkpoint.save_dir=$LOG_DIR/ckpt \
    common.log_file=$LOG_DIR/log.txt \
    common.tensorboard_logdir=$LOG_DIR/tb \
    distributed_training.distributed_world_size=$WORLD_SIZE