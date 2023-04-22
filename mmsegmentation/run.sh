GPU=$2
port=23466


# config=swin-base-patch4-window7_upernet_8xb2-160k_fault-512x512
config=swin-base-patch4-window7_upernet_8xb2-160k_fault_public-512x512

if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PORT=${port} ./tools/dist_train.sh ./projects/Fault_recong/config/${config}.py 1 --work-dir output/${config} 
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh configs/swin/${config}.py ./cache/${config}/best_mDice_epoch_20.pth 1 --format-only --work-dir cache/${config} --eval-options "imgfile_prefix=./test_results/${config}"
fi