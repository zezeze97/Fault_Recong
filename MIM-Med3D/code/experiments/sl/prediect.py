from multi_seg_main import MultiSegtrainer
import yaml
from pytorch_lightning import Trainer
from data.Fault_dataset import FaultDataset
import os
import numpy as np
import h5py
import shutil


def predict(config_path, ckpt_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(config_path,encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    model = MultiSegtrainer(**config['model']['init_args'])
    trainer = Trainer(accelerator='gpu', devices=-1, default_root_dir=output_path, inference_mode=True)
    dataloader = FaultDataset(**config['data']['init_args'])
    dataloader.setup(stage='test')
    pred_loader = dataloader.test_dataloader()
    preds = trainer.predict(model, pred_loader,ckpt_path=ckpt_path)
    for item in preds:
        for image_name in item.keys():
            pred = item[image_name]
            pred = pred.squeeze(0) # (128, 128, 128)
            with h5py.File(os.path.join(output_path, image_name), 'w') as f:
                f['predictions'] = pred.cpu().numpy()
    shutil.rmtree(os.path.join(output_path, 'lightning_logs'))
    # print(preds)
        
        


if __name__ == '__main__':
    config_path = './code/configs/sl/fault/unetr_base_supbaseline.yaml'
    ckpt_path = './output/Fault_Baseline/unetr_base_supbaseline_p16_fault/checkpoints/best.ckpt'
    output_path = './output/Fault_Baseline/unetr_base_supbaseline_p16_fault/preds'
    predict(config_path, ckpt_path, output_path)