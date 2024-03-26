#!/bin/bash
GPU=$1
NUM_OF_GPU=$2
port=23509


config=swin-base-simmim
# config=swin-base-simmim-yd-data
CUDA_VISIBLE_DEVICES=$GPU PORT=${port} ./tools/dist_train.sh ./projects/Fault_recong/config/${config}.py ${NUM_OF_GPU} --work-dir output/${config}
