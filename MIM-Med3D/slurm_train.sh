#!/bin/bash -l
#SBATCH -o job.%j.out
#SBATCH --partition=GPU40G
#SBATCH --qos=low
#SBATCH -J swin_unetr_base_seg_cls_simmim_2x4
#SBATCH --nodes=2          
#SBATCH --cpus-per-task=8   
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00

source activate $1
MAIN_FILE=$2
CONFIG_FILE=$3
srun python3 $MAIN_FILE fit --config $CONFIG_FILE
