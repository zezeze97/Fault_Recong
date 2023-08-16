#!/bin/bash -l
#SBATCH -o job.%j.out
#SBATCH --partition=GPU40G
#SBATCH --qos=high
#SBATCH -J swin_unetr_simmim_v2_4x4
#SBATCH --nodes=4          
#SBATCH --cpus-per-task=16   
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=18428308691@163.com

source activate $1
MAIN_FILE=$2
CONFIG_FILE=$3
srun python3 $MAIN_FILE fit --config $CONFIG_FILE
