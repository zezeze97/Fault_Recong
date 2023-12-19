# Fault Recognition Based on Transformer

# Codebase Overview
- Developed entirely using the PyTorch deep learning framework.
- The 2D model relies on [mmcv](https://github.com/open-mmlab/mmcv), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), and [mmpretrain](https://github.com/open-mmlab/mmpretrain).
- The 3D model is built on [pytorch-lightning](https://github.com/Lightning-AI/lightning).
- 2D segmentation model code: [mmsegmentation](./mmsegmentation/), with project-related configuration files located at [mmsegmentation/projects/Fault_recong](./mmsegmentation/projects/Fault_recong).
- 2D pretraining code: [mmpretrain](./mmpretrain/), with project-related configuration files located at [mmpretrain/projects/Fault_Recong](./mmpretrain/projects/Fault_Recong).
- [3D segmentation and pretraining code](./MIM-Med3D/), with project-related configuration files located at [./MIM-Med3D/code/configs](./MIM-Med3D/code/configs).


# Environment Setup
The codes for 2D segmentation, pretraining, and 3D segmentation/pretraining are independent. If using only the 2D model, install the necessary environment for the 2D model. Since the entire project is developed in PyTorch, begin by installing PyTorch. I am using PyTorch version 1.12.1+cu113. Due to potential differences in CUDA versions, the torch installation may vary. You can directly install torch from the [official website](https://pytorch.org/get-started/pytorch-2.0/). I recommend installing torch-1.12.1 version. After torch installation, proceed with configuring the environment for either the 2D or 3D models.

For instance, on a machine with CUDA version 11.3, installing torch 1.12.1 is done using the following command from the torch official website:

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
## 2D Segmentation Model Environment Setup

Navigate to the [2D segmentation codebase](./mmsegmentation). The original documentation for the codebase is found at [./mmsegmentation/README_zh-CN.md](./mmsegmentation/README_zh-CN.md). Here are simplified installation steps. If issues arise, refer to the official mmsegmentation documentation.

Step 0: Use MIM to install MMCV
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
if MIM installation is not possible, visit the [MMCV official website](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html), choose the appropriate torch and CUDA versions, and install using pip. For example, installing mmcv based on CUDA 11.3 and torch 1.12.x is done with:
```
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```
Step 1: Install MMSegmentation
```
cd mmsegmentation
pip install -v -e .
# '-v' indicates verbose mode for more output
# '-e' indicates editable mode installation, so any changes to the code take effect without reinstallation

```

## 2D Self-Supervised Pretraining Environment Setup
Navigate to the [2D self-supervised pretraining codebase](./mmpretrain). The original documentation for the codebase is found at [./mmpretrain/README_zh-CN.md](./mmpretrain/README_zh-CN.md). Here are simplified installation steps. If issues arise, refer to the official mmpretrain documentation.
```
cd mmpretrain
pip3 install openmim
mim install -e .
```

## 3D Segmentation Model Environment Setup
Navigate to the [3D segmentation codebase](./MIM-Med3D):
```
cd MIM-Med3D
pip install -r requirements.txt
```

# 2D Model Prediction Interface
In the [2D segmentation folder](./mmsegmentation/)

The general format is to call the predict_3d or predict_2d function in [./mmsegmentation/projects/Fault_recong/predict.py](./mmsegmentation/projects/Fault_recong/predict.py).

The predict_3d function accepts input in .npy or .sgy format.

The predict_2d function accepts input as a folder containing all the 2D images to be predicted, and the files inside the folder should be .npy or .png single-channel images.
```
cd mmsegmentation
python ./projects/Fault_recong/predict.py --config {Path to model config} \
                                        --checkpoint {Model checkpoint path} \
                                        --input {Input image root dir/cube path} \
                                        --save_path {Path to save predict result} \
                                        --predict_type {Predict 2d/3d fault} \
                                        --convert_25d {Whether convert to 2.5d} \ 
                                        --step {step size of 2.5d data} \ 
                                        --force_3_chan {Whether convert to 3 channel} \
                                        --device {Set cuda device} \
                                        --direction {inline/xline} \
```
Additionally, [./mmsegmentation/projects/Fault_recong/predict.py](./mmsegmentation/projects/Fault_recong/predict.py) provides the predict_2d_single_image function, supporting numpy arrays as input. Other parameters are the same as predict_2d and can be used for single image prediction. The function returns the predicted fault score for the input image.
```
# Example usage
input = np.load(f'{image_path}')  # Single-channel image
config_file = './output/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e.py'
checkpoint_file = './output/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e/best.pth'
device = 'cuda:0'
score = predict_2d_single_image(config_file, checkpoint_file, input, device, force_3_chan=True)  # score is the predicted fault score for the image [0,1]
```

## 2D Network Trained with Mixed Data
To predict new data using a 2D network trained with mixed data, use the following command:
```
cd mmsegmentation
sh predict.sh {Input cube(*.npy/*.sgy)} {Save path}
```

# 2D Model Training Interface

Invoke the configuration file [./mmsegmentation/projects/Fault_recong/config/swin-base-simmim.py](./mmsegmentation/projects/Fault_recong/config/swin-base-simmim.py). Note the specification of the data_root on line 71, where the data structure should be as follows:
```
.
├── train
│   ├── ann
│   └── image
└── val
    ├── ann
    └── image

```
The ann folder contains *.png annotations for (0,1) fault layers, and the image folder contains *.npy data bodies. The filenames in image and ann should correspond.
```
cd mmsegmentation
# bash run.sh {GPU_NUM}
# e.g., using a single GPU, starting from self-supervised pretraining checkpoint, training for 16000 iterations, and saving checkpoints in ./output/swin-base-simmim folder
sh run.sh 0
```

# 3D Model Prediction Interface
In the [3D model codebase](./MIM-Med3D/), call the predict_sliding_window function in [./MIM-Med3D/code/experiments/sl/predict.py](./MIM-Med3D/code/experiments/sl/predict.py). The model will perform slice inference on 3D faults with a size of 128x128x128. The function accepts input in .npy or .sgy format. The generic format for the call is as follows:
```
python ./code/experiments/sl/prediect.py --config {Path to model config} \
                                        --checkpoint {Model checkpoint path} \
                                        --input {Input cube path} \
                                        --save_path {Path to save predict result} \
                                        --device {Set cuda device} \
```
The prediction results and scores for each pixel will be saved in the save_path folder.

## 3D Segmentation Model Trained on Thebe Data
```
cd MIM-Med3D
sh predict.sh {Input cube(*.npy/*.sgy)} {Save path}
```

# 3D Model Training Interface
Model training is initiated using [./code/experiments/sl/multi_seg_main.py](./code/experiments/sl/multi_seg_main.py), with the configuration file [./code/configs/sl/fault/swin_unetr_ft.yaml](./code/configs/sl/fault/swin_unetr_ft.yaml). Specify the data location in the labeled_data_root_dir_lst on line 109. The required data structure under data_root is as follows:
```
.
├── train
└── val
```
Folders train and val contain a series of *.h5 files, each file containing "raw" and "label" corresponding to segmented 256 * 256 * 256 size data bodies and fault layers.
```
# Starting from self-supervised pretraining checkpoint, perform fine-tuning for 1000 epochs. The training results are saved in ./output/Fault_Finetuning/swin_unetr_ft folder.
sh train.sh ./code/experiments/sl/multi_seg_main.py ./code/configs/sl/fault/swin_unetr_ft.yaml
```
# SAM Adapter FT using Large Model
Utilize the separate repository [SAM-Adapter](./SAM-Adapter/). The original documentation for this repository is found at [SAM-Adapter/README.md](SAM-Adapter/README.md).

Create a virtual environment for SAM Adapter:
```
cd SAM-Adapter
conda env create -f environment.yml
conda activate sam_adapt

# Fine-tune 2D data. Checkpoints are saved in the ./SAM-Adapter/logs folder.
python train.py -net sam -mod sam_adpt -exp_name Fault2D_SAM -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 4 -dataset fault2d -data_path ../Fault_data/public_data/2d_slices -val_freq 1 -vis 50

# Predict
python predict.py --device cuda:0 --input_cube_path {input .npy/.sgy path} --save_path {Path to save score} --sam_ckpt {ft ckpt path}

```