import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning, BackboneFinetuning
from torch.optim.optimizer import Optimizer

class SwinTransformerFinetuner(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=100) -> None:
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
    
    def freeze_before_training(self, pl_module) -> None:
        self.freeze(pl_module.model.swinViT)
    
    def finetune_function(self, pl_module, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        if epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                    modules=pl_module.model.swinViT,
                    optimizer=optimizer,
                    train_bn=True,
                    )