# 基于Transformer的断层识别

# 代码库简介

 - 全部使用pytorch深度学习框架
 - 2D模型基于[mmcv](https://github.com/open-mmlab/mmcv), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), 以及[mmpretrain](https://github.com/open-mmlab/mmpretrain).
 - 3D模型基于[pytorch-lightning](https://github.com/Lightning-AI/lightning)开发
 - 2D分割模型代码: [mmsegmentation](./mmsegmentation/), 项目相关配置文件位于[mmsegmentation/projects/Fault_recong](./mmsegmentation/projects/Fault_recong)
 - 2D预训练代码: [mmpretrain](./mmpretrain/), 项目相关配置文件位于[mmpretrain/projects/Fault_Recong](./mmpretrain/projects/Fault_Recong)
 - [3D分割, 预训练代码](./MIM-Med3D/), 项目相关配置文件位于[./MIM-Med3D/code/configs](./MIM-Med3D/code/configs)

# 环境安装
2D分割, 预训练, 3D分割、预训练代码相互独立, 如只需使用2D模型, 仅需要安装2D模型需要的环境即可. 由于整个项目是由pytorch开发, 首先需要安装pytorch, 我使用的是PyTorch: 1.12.1+cu113版本的torch, 由于CUDA版本不同, 可能需要安装的torch略有差异, 可直接去[官网](https://pytorch.org/get-started/pytorch-2.0/)安装torch. 推荐安装torch-1.12.1版本. **在安装完torch之后, 才可以进行2D或者3D模型的环境配置**

例如在cuda版本为11.3的机器上安装torch 1.12.1, torch官网给出的安装命令为
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## 2D分割模型环境安装
进入[2D分割代码库](./mmsegmentation), 原本代码库的说明文档位于[./mmsegmentation/README_zh-CN.md](./mmsegmentation/README_zh-CN.md), 这里简单说明一下安装步骤, 如遇问题可参考mmsegmentation的官方文档.
 
步骤0: 使用MIM安装MMCV
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
如果无法使用MIM安装, 可去[MMCV官网](https://mmcv.readthedocs.io/zh_CN/latest/get_started/installation.html), 选择合适的torch和cuda版本, 使用pip安装. 例如安装基于cuda11.3, torch1.12.x的mmcv命令为
```
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```

步骤1: 安装MMSegmentation
```
cd mmsegmentation
pip install -v -e .
# '-v' 表示详细模式，更多的输出
# '-e' 表示以可编辑模式安装工程，
# 因此对代码所做的任何修改都生效，无需重新安装
```
## 2D自监督预训练环境安装
进入[2D自监督预训练代码库](./mmpretrain), 原本代码库的说明文档位于[./mmpretrain/README_zh-CN.md](./mmpretrain/README_zh-CN.md), 这里简单说明一下安装步骤, 如遇问题可参考mmpretrain的官方文档.
```
cd mmpretrain
pip3 install openmim
mim install -e .
```

## 3D分割模型环境安装
进入[3D分割代码库](./MIM-Med3D):
```
cd MIM-Med3D
pip install -r requirements.txt
```


# 2D模型预测接口
在[2D分割文件夹下](./mmsegmentation)

通用格式, 调用[./mmsegmentation/projects/Fault_recong/predict.py](./mmsegmentation/projects/Fault_recong/predict.py)中的predict_3d或predict_2d函数. 

其中predict_3d函数接受的输入为.npy或者.sgy文件

predict_2d函数接受的输入为包含所有需要预测的2d图片的文件夹, 里面的文件为.npy或者.png的单通道图片. 
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
除此之外, 在[./mmsegmentation/projects/Fault_recong/predict.py](./mmsegmentation/projects/Fault_recong/predict.py)中还提供了predict_2d_single_image函数, 支持numpy数组作为输入, 其余参数与predict_2d函数相同, 可用于单张图片的预测, 函数返回的是输入图片的断层预测得分
```
# 使用示例
input = np.load(f'{image_path}') # 单通道图片
config_file = './output/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e.py'
checkpoint_file = './output/swin-base-patch4-window7_upernet_8xb2-160k_mix_data_v3_force_3_chan-512x512_per_image_normal_simmim_2000e/best.pth'
device = 'cuda:0'
score = predict_2d_single_image(config_file, checkpoint_file, input, device, force_3_chan=True) # score为该图片的断层预测得分[0,1]


```
## 使用混合数据训练的2D网络
可以使用如下命令调用混合数据训练的2D网络模型对新的数据进行预测
```
cd mmsegmentation
sh predict.sh {Input cube(*.npy/*.sgy)} {Save path}
```

# 2D模型训练接口

调用的配置文件为[./mmsegmentation/projects/Fault_recong/config/swin-base-simmim.py](./mmsegmentation/projects/Fault_recong/config/swin-base-simmim.py), 注意指定第71行的data_root, 数据格式要求为./data_root下包含的文件结构为
```
.
├── train
│   ├── ann
│   └── image
└── val
    ├── ann
    └── image

```
其中ann文件夹下为*.png的(0,1)体断层标注, image文件夹下为*.npy的数据体, image和ann的文件名需要对应上.
```
cd mmsegmentation
# bash run.sh {GPU_NUM}
# eg 使用单卡, 从自监督预训练ckpt开始, 训练16000个iter, 训练的ckpts保存在./output/swin-base-simmim文件夹下
sh run.sh 0
```

# 3D模型预测接口
在[3D模型代码库](./MIM-Med3D/)下, 调用[./MIM-Med3D/code/experiments/sl/predict.py](./MIM-Med3D/code/experiments/sl/predict.py)中的predict_sliding_window函数, 模型会按照128x128x128的大小对输入的3D断层进行slice inferrence. 该函数接受的输入为.npy或者.sgy文件, 调用的通用格式如下
```
python ./code/experiments/sl/prediect.py --config {Path to model config} \
                                        --checkpoint {Model checkpoint path} \
                                        --input {Input cube path} \
                                        --save_path {Path to save predict result} \
                                        --device {Set cuda device} \
```
预测的结果以及每个像素点的得分会保存在save_path文件夹下..

## 基于Thebe数据训练的3D分割模型
```
cd MIM-Med3D
sh predict.sh {Input cube(*.npy/*.sgy)} {Save path}
```

# 3D模型训练接口
模型的训练调用[./code/experiments/sl/multi_seg_main.py](./code/experiments/sl/multi_seg_main.py), 配置文件为[./code/configs/sl/fault/swin_unetr_ft.yaml](./code/configs/sl/fault/swin_unetr_ft.yaml), 需要在109行的labeled_data_root_dir_lst指定数据的位置, 数据格式要求为data_root下包含的文件结构为
```
.
├── train
└── val
```
train/val文件夹下为一系列*.h5文件, 每个文件内包含"raw", "label"分别对应切割好的256 * 256 *256 大小的数据体，断层体
```
# 从自监督预训练ckpt开始, 进行1000个epoch的ft, 训练结果保存在./output/Fault_Finetuning/swin_unetr_ft文件夹下
sh train.sh ./code/experiments/sl/multi_seg_main.py ./code/configs/sl/fault/swin_unetr_ft.yaml
```
