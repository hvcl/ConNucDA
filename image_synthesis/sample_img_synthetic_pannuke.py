# from share import *
# import config

import os
# import cv2
import einops
# import gradio as gr
import numpy as np
import torch
import random
import torchvision

from PIL import Image
# from tqdm import tqdm
from seg_edge_hv.sample_synthetic.pannuke_dataset_test import MyDataset
# from tutorial_dataset_test_lizard import MyDataset
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
# from annotator.util import resize_image, HWC3
# from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from collections import namedtuple

PanukeClass = namedtuple('pannukeClass', ['name', 'id', 'has_instances', 'color'])
# autopep8: off
classes = [
    PanukeClass('Background',                0, True,  (  0,   0,   0)), # black
    PanukeClass('Inflammatory',              1, True,  (250, 170, 160)), # peach
    PanukeClass('Connective_tissue',         2, True,  (164, 164, 238)), # light purple
    PanukeClass('Dead',                      3, True,  ( 51,  51, 255)), # blue
    PanukeClass('non-neoplastic epithelial', 4, True,  (231, 205,  13)), # yellow
    PanukeClass('neoplastic',                5, True,  ( 76, 153,   0)), # green / Non_Neoplastic_Epithelial
    PanukeClass('edge',                      6, True,  (255, 255, 255)), # white
]
# autopep8: on
num_classes = 7
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
        if c>6:
            continue
        lbl_3ch[lbl[:,c]==1] = c # [B,H,W]
    lbl_3ch = lbl_3ch.long()
    # print("colors[lbl_3ch].shape: ", colors[lbl_3ch].shape)
    # exit()
    rgbs = colors[lbl_3ch]
    rgbs = rgbs.permute(0, 3, 1, 2)
    return rgbs / 255.


batch_size = 15
seed = random.randint(0, 65535)
seed_everything(seed)

model = create_model('./models/cond_seg_edge_hv/cldm_pathLDM_pannuke.yaml').cpu()

### seg edge hv
model.load_state_dict(load_state_dict('./lightning_logs/pannuke-setting_iter100k_b32/steps26.4k+50k/epoch=249-step=49999.ckpt', location='cuda'))

model = model.cuda()
ddim_sampler = DDIMSampler(model)

dataset = MyDataset(syn_set="test_set_final")
# test_set_final
# test_set_1
# test_set_2
# test_set_3
# test_set_4
# val_set_1
# val_set_2
# val_set_3
# val_set_4

dataloader = DataLoader(dataset, num_workers=5, batch_size=batch_size, shuffle=False)

# DDIM setting
ddim_steps = 100
scale = 5.5
eta = 0.0
# scales = [1.5, 2.0, 2.5, 3.0, 3.5]
##############################
# scales = [2.5, 3.0, 3.5]
# scales = [4.0, 4.5, 5.0]
# scales = [5.5, 6.0, 6.5]
# scales = [7.0, 7.5, 8.0]
# scales = [8.5, 9.0, 9.5]
# scales = [10.0, 10.5, 11.0]
##############################

# data_path = "/Dataset/lizard/NASDM/lizard_split_norm_256/MICCAI2024_synthetic"
data_path = dataset.pannuke_data_path
# print(data_path)
# exit()

# for scale in scales:
sample_dir = os.path.join(data_path, "images")
grid_dir = os.path.join(data_path, "grids")
# sample_dir = f"./lightning_logs/lizard-setting_iter100k_b32/samples_synthetic/scale{scale}"
# grid_dir = f"./lightning_logs/lizard-setting_iter100k_b32/grids_synthetic/scale{scale}"
# sample_dir = f"./lightning_logs/lizard-setting_iter100k_b32/samples/scale{scale}"
# grid_dir = f"./lightning_logs/lizard-setting_iter100k_b32/grids/scale{scale}"
# sample_dir = f"./lightning_logs/lizard-setting_iter50k_b32/samples/scale{scale}"
# grid_dir = f"./lightning_logs/lizard-setting_iter50k_b32/grids/scale{scale}"
# sample_dir = f"./lightning_logs/lizard-setting_iter50k_b32/samples_synthetic/scale{scale}"
# grid_dir = f"./lightning_logs/lizard-setting_iter50k_b32/grids_synthetic/scale{scale}"

os.makedirs(sample_dir, exist_ok=True)
os.makedirs(grid_dir, exist_ok=True)

idx_batch = 0
print("currently working on:",data_path)
for batch in dataloader:

    cur_batch_size = len(batch['txt'])

    orig_imgs = batch['orig'].permute(0, 3, 1, 2)
    imgs = batch['img'].permute(0, 3, 1, 2)
    fn_exts = batch['fn_ext']
    prompt = batch['txt']
    control = batch['hint'] # []

    control = control.permute(0, 3, 1, 2) # [B, C, H, W]
    control_cuda = control.cuda()
    
    cond = {
        "c_concat": [control_cuda],
        "c_crossattn": [model.get_learned_conditioning(prompt)]
    }
    
    shape = (3, 64, 64)

    samples, intermediates = ddim_sampler.sample(ddim_steps, cur_batch_size,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale)
                                                    # unconditional_conditioning=un_cond)    
    cond_lbl = transform_lbl(control) # [B, 3, H, W] 0~1
    x_samples = model.decode_first_stage(samples)
    x_samples_0to1 = ((x_samples + 1) / 2).clip(0, 1)
    x_samples_uint8 = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    # print(batch.keys()) # txt, hint
    # exit()

    # Save images
    for fn_ext, x_sample in zip(fn_exts, x_samples_uint8):
        Image.fromarray(x_sample).save(os.path.join(sample_dir, fn_ext))

    # Grid: [orig / sampled / cond]
    grid_5 = orig_imgs.shape[0]//5
    for b_grid in range(grid_5):
        img_cur_grid = orig_imgs[b_grid*5:(b_grid+1)*5]
        sam_cur_grid = x_samples_0to1[b_grid*5:(b_grid+1)*5].cpu()
        lbl_cur_grid = cond_lbl[b_grid*5:(b_grid+1)*5]

        grid_img = torchvision.utils.make_grid(img_cur_grid, nrow=5) # [3, 260, 1286]
        grid_sam = torchvision.utils.make_grid(sam_cur_grid, nrow=5) # [3, 260, 1286]
        grid_lbl = torchvision.utils.make_grid(lbl_cur_grid, nrow=5) # [3, 260, 1286]
        
        grid_img = grid_img.transpose(0, 1).transpose(1, 2) #.squeeze(-1) # [260, 1286, 3]
        grid_sam = grid_sam.transpose(0, 1).transpose(1, 2) #.squeeze(-1) # [260, 1286, 3]
        grid_lbl = grid_lbl.transpose(0, 1).transpose(1, 2) #.squeeze(-1) # [260, 1286, 3]

        # print(torch.max(grid_sam), torch.min(grid_sam))
        # exit()

        grid_img = (grid_img.numpy() * 255).clip(0, 255).astype(np.uint8)
        grid_sam = (grid_sam.numpy() * 255).clip(0, 255).astype(np.uint8)
        grid_lbl = (grid_lbl.numpy() * 255).clip(0, 255).astype(np.uint8)

        # print(grid_img.shape, grid_sam.shape, grid_lbl.shape)

        grid = np.concatenate([grid_img, grid_sam, grid_lbl], axis=0)

        Image.fromarray(grid).save(os.path.join(grid_dir, f"grid_{idx_batch:03d}.png"))

        idx_batch += 1

    if grid_5==0:
        grid_img = torchvision.utils.make_grid(orig_imgs, nrow=5) # [3, 260, 1286]
        grid_sam = torchvision.utils.make_grid(x_samples_0to1.cpu(), nrow=5) # [3, 260, 1286]
        grid_lbl = torchvision.utils.make_grid(cond_lbl, nrow=5) # [3, 260, 1286]
        
        grid_img = grid_img.transpose(0, 1).transpose(1, 2) #.squeeze(-1) # [260, 1286, 3]
        grid_sam = grid_sam.transpose(0, 1).transpose(1, 2) #.squeeze(-1) # [260, 1286, 3]
        grid_lbl = grid_lbl.transpose(0, 1).transpose(1, 2) #.squeeze(-1) # [260, 1286, 3]

        # print(torch.max(grid_sam), torch.min(grid_sam))
        # exit()

        grid_img = (grid_img.numpy() * 255).clip(0, 255).astype(np.uint8)
        grid_sam = (grid_sam.numpy() * 255).clip(0, 255).astype(np.uint8)
        grid_lbl = (grid_lbl.numpy() * 255).clip(0, 255).astype(np.uint8)

        # print(grid_img.shape, grid_sam.shape, grid_lbl.shape)

        grid = np.concatenate([grid_img, grid_sam, grid_lbl], axis=0)

        Image.fromarray(grid).save(os.path.join(grid_dir, f"grid_{idx_batch:03d}.png"))

        idx_batch += 1

    # print(sample_dir, 'scale:', scale)
