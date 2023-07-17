#!/bin/bash -l
#SBATCH -o job.%j.out
#SBATCH --partition=GPU
#SBATCH --qos=low
#SBATCH -J SimMIM_Swin_4x8
#SBATCH --nodes=4          
#SBATCH --cpus-per-task=2   
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8  
#SBATCH --time=5-00:00:00

export NCCL_P2P_DISABLE=1
source activate $1
CONFIG=$2
WORK_DIR=$3
srun python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm"