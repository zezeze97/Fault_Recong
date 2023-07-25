from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
# from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete, EnsureType
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import sys
sys.path.insert(0,'./code')
from models import SwinUNETR_Seg_Cls
from metrics import dice_coefficient_batch, compute_acc_batch, compute_f1_batch
import optimizers
import data

class SegCls_trainer(pl.LightningModule):
    def __init__(
        self, model_name: str, model_dict: dict,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dict = model_dict

        if model_name == 'swin_unetr_seg_cls':
            self.model = SwinUNETR_Seg_Cls(**model_dict)
        else:
            raise NotImplementedError

        # self.loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, lambda_dice=0.1, lambda_ce=1.0)
        self.seg_loss_function = torch.nn.BCEWithLogitsLoss()
        self.cls_loss_function = torch.nn.BCEWithLogitsLoss()
        # self.loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10))
        # self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        # self.post_label = AsDiscrete(to_onehot=num_classes)
        self.post_trans = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )
        self.post_score_trans = Compose([EnsureType(), Activations(sigmoid=True)])
        self.f1_vals = []
        # self.dice_vals_tc = []
        # self.dice_vals_wt = []
        # self.dice_vals_et = []
        self.metric_values = []
        self.epoch_loss_values = []

    def forward(self, x):
        return self.model(x)
    
    def forward_test(self, x):
        seg_logits, cls_logits = self.model(x)
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            if self.post_score_trans(cls_logits[i,:]) < 0.5:
                seg_logits[i, :, :, :, :] = -1000.0 * torch.ones((seg_logits[i, :, :, :, :].shape), device=seg_logits.device)
        return seg_logits

    def generate_label_cls(self, labels_seg):
        batch_size = labels_seg.shape[0]
        labels_cls = torch.zeros((batch_size, 1), dtype=torch.float, device=labels_seg.device)
        for i in range(batch_size):
            if torch.sum(labels_seg[i, :, :, :, :]) >= 0.0001 * labels_seg.shape[2] * labels_seg.shape[3] * labels_seg.shape[4]:
                labels_cls[i, :] = 1.0
        return labels_cls


    def training_step(self, batch, batch_idx):
        images, labels_seg = batch["image"], batch["label"]
        batch_size = images.shape[0]
        labels_cls = self.generate_label_cls(labels_seg)
        seg_logits, cls_logits = self.forward(images)
        seg_loss = self.seg_loss_function(seg_logits, labels_seg)
        cls_loss = self.cls_loss_function(cls_logits, labels_cls)
        # logging
        self.log(
            "train/seg_bce_loss_step",
            seg_loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            "train/cls_bce_loss_step",
            cls_loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        loss = seg_loss + cls_loss
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train/bce_loss_avg",
            avg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels_seg = batch["image"], batch["label"]
        batch_size = images.shape[0]
        
        roi_size = (128, 128, 128)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward_test,  # the output image will be cropped to the original image size
        )
        
        # outputs = self.forward(images)
        
        seg_loss = self.seg_loss_function(outputs, labels_seg)
        # compute dice score
        outputs = [self.post_trans(i).detach().cpu().numpy() for i in decollate_batch(outputs)]
        labels_seg = [label.detach().cpu().numpy() for label in decollate_batch(labels_seg)]
        f1_batch = compute_f1_batch(labels_seg, outputs)
        self.f1_vals += f1_batch
        # logging
        self.log(
            "val/seg_bce_loss_step",
            seg_loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        return {"val_seg_loss": seg_loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_seg_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_f1 = np.mean(self.f1_vals)
        self.f1_vals = []
        mean_val_loss = torch.tensor(val_loss / num_items)
        # logging
        self.log(
            "val/seg_bce_loss_avg",
            mean_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            "val/f1_avg",
            mean_val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.logger.log_hyperparams(
            params={
                "model": self.model_name,
                **self.model_dict,
                # "data": self.trainer.datamodule.json_path,
                # "ds_ratio": self.trainer.datamodule.downsample_ratio,
                "batch_size": self.trainer.datamodule.batch_size,
                "distribution": self.trainer.datamodule.dist,
                # "benchmark": self.trainer.benchmark,
                "max_epochs": self.trainer.max_epochs,
                "precision": self.trainer.precision,
            },
            metrics={"bce_loss": mean_val_loss, "f1": mean_val_f1},
        )

        self.metric_values.append(mean_val_f1)

    def test_step(self, batch, batch_idx):
        images, labels_seg = batch["image"], batch["label"]
        
        batch_size = images.shape[0]
        roi_size = (128, 128, 128)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward_test,  # the output image will be cropped to the original image size
        )
       
        loss = self.seg_loss_function(outputs, labels_seg)
        # compute dice score
        outputs = [self.post_trans(i).detach().cpu().numpy() for i in decollate_batch(outputs)]
        labels_seg = [label.detach().cpu().numpy() for label in decollate_batch(labels_seg)]
        f1_batch = compute_f1_batch(labels_seg, outputs)
        # print(dice_batch)

        return {"f1_batch": f1_batch}

    def test_epoch_end(self, outputs):
        f1_vals = []
        for output in outputs:
             f1_vals += output["f1_batch"]
        mean_val_f1 = np.mean(f1_vals)

        print(f"avg f1: {mean_val_f1} ")
    
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        images = batch["image"]
        image_names = batch['image_name']

        roi_size = (128, 128, 128)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward_test,  # the output image will be cropped to the original image size
            # device=torch.device('cpu'),
            # progress=True
        )
        outputs_pred = {}
        for i, output in enumerate(decollate_batch(outputs)):
            score = self.post_score_trans(output)
            pred = self.post_trans(output)
            outputs_pred[image_names[i]] = {'pred': pred, 'score': score}

        return outputs_pred
        


if __name__ == "__main__":
    cli = LightningCLI(save_config_kwargs={'overwrite': True})
