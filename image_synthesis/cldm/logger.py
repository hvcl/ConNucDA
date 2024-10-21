import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from collections import namedtuple

LizardClass = namedtuple('LizardClass', ['name', 'id', 'has_instances', 'color'])
# autopep8: off
classes = [
    LizardClass('background',         0, True,  (  0,   0,   0)), # black
    LizardClass('Neutrophil',         1, True,  (204,   0,   0)), # red
    LizardClass('Epithelial',         2, True,  ( 76, 153,   0)), # green
    LizardClass('Lymphocyte',         3, True,  (255, 153,  51)), # orange
    LizardClass('Plasma',             4, True,  (153,  76, 255)), # purple
    LizardClass('Neutrophil',         5, True,  (255,  51, 153)), # pink
    LizardClass('Connective_tissue',  6, True,  ( 51,  51, 255)), # cyan
    LizardClass('edge',               7, True,  (255, 255, 255)), # white
]
# autopep8: on
num_classes = 8
mapping_id = torch.tensor([x.id for x in classes])
colors = torch.tensor([cls.color for cls in classes])

def transform_lbl(lbl: torch.Tensor, *args, **kwargs):
    lbl = lbl.long()
    if lbl.size(1) == 1:
        # Remove single channel axis.
        lbl = lbl[:, 0]
    B, C, H, W = lbl.shape
    lbl_3ch = torch.zeros([B, H, W])
    for c in range(C):
        lbl_3ch[lbl[:,c]==1] = c # [B,H,W]
    lbl_3ch = lbl_3ch.long()
    # print("colors[lbl_3ch].shape: ", colors[lbl_3ch].shape)
    # exit()
    rgbs = colors[lbl_3ch]
    rgbs = rgbs.permute(0, 3, 1, 2)
    return rgbs / 255.


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        # Dict of images: 
        # dict_keys(['reconstruction', 'control', 'conditioning', 'samples_cfg_scale_9.00'])
        
        grid_all = {}
        for idx, k in enumerate(images):
            if k=='control':
                images[k] = transform_lbl(images[k])
            # if k=='conditioning':
            #     B, C, H, W = images[k].shape
            #     images[k] = images[k].view([B, C, 256, 256])
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            # 1. Control to 3ch
            # if k=='control':
            #     print(grid.shape)
            #     transform_lbl(grid) # c,h,w -> 
            #     exit()
            # # 2. Conditioning: 512x512 -> 256x256?
            # # 3. -
            # else:
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            grid_all[k] = grid
        grid = np.concatenate(
            [
                grid_all['reconstruction'],
                grid_all['control'],
                grid_all['conditioning'],
                grid_all['samples_cfg_scale_2.50']
            ], axis=0
        )
        # filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
        filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        print('grid shape:', grid.shape)
        Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            
            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            return
            self.log_img(pl_module, batch, batch_idx, split="train")
