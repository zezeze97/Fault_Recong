#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPU40G
#SBATCH --qos=low
#SBATCH -J swin_unetr_base_multi_decoder
#SBATCH --nodes=1          
#SBATCH --cpus-per-task=8   
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4     
#SBATCH --time=5-00:00:00


MAIN_FILE=$1
CONFIG_FILE=$2
srun python3 $MAIN_FILE fit --config $CONFIG_FILE
