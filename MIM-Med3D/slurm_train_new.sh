#!/usr/bin/env bash

MAIN_FILE=$1
CONFIG_FILE=$2
srun --partition=GPU40G \
    --qos=low \
    -J swin_unetr_base_multi_decoder \
    --nodes=1 \
    --cpus-per-task=8 \
    --ntasks-per-node=4 \
    --gres=gpu:4 \
    --time=5-00:00:00 \
    python3 $MAIN_FILE fit --config $CONFIG_FILE \
    2>&1 | tee train.log
