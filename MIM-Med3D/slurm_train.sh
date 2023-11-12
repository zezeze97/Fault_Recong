#!/bin/bash -l
#SBATCH -o job.%j.out
#SBATCH --partition=GPU40G
#SBATCH --qos=high
#SBATCH -J swin_unetr_base_multi_decoder_fusion_overall_simmim300e_p16_public_256_4x4_rerun
#SBATCH --nodes=4          
#SBATCH --cpus-per-task=16   
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4  
#SBATCH --time=5-00:00:00

source activate $1
MAIN_FILE=$2
CONFIG_FILE=$3
srun python3 $MAIN_FILE fit --config $CONFIG_FILE
