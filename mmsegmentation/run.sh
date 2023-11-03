#!/bin/bash
GPU=$1
port=23509


config=swin-base-simmim
CUDA_VISIBLE_DEVICES=$GPU PORT=${port} python ./tools/train.py ./projects/Fault_recong/config/${config}.py --work-dir output/${config}
