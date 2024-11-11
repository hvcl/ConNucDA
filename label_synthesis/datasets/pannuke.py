import json
import os
import os.path as osp
import random
from collections import namedtuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor

PanukeClass = namedtuple('PanukeClass', ['name', 'id', 'has_instances', 'color'])
# autopep8: off
classes = [
    PanukeClass('Background',         0, True,  (  0,   0,   0)), # black
    PanukeClass('Neoplastic',         1, True,  (250, 170, 160)), # peach
    PanukeClass('Inflammatory',       2, True,  (164, 164, 238)), # light purple
    PanukeClass('Connective_tissue',  3, True,  ( 51,  51, 255)), # blue
    PanukeClass('Dead',               4, True,  (231, 205,  13)), # yellow
    PanukeClass('Epithelial',         5, True,  ( 76, 153,   0)), # green / Non_Neoplastic_Epithelial
]
# classes = [
#     PanukeClass('Background',         0, True,  (  0,   0,   0)), # black
#     PanukeClass('Neoplastic',         1, True,  (250, 170, 160)), # peach (250, 170, 160)
#     PanukeClass('Inflammatory',       2, True,  (  0, 244, 238)), # cyan
#     PanukeClass('Connective_tissue',  3, True,  ( 51,  51, 255)), # blue
#     PanukeClass('Dead',               4, True,  (248, 242,   0)), # yellow
#     PanukeClass('Epithelial',         5, True,  ( 76, 153,   0)), # green / Non_Neoplastic_Epithelial
# ]
# autopep8: on
num_classes = 6
mapping_id = torch.tensor([x.id for x in classes])
colors = torch.tensor([cls.color for cls in classes])
# PANNUKE_CLASSES = ['', 'neoplastic', 'inflammatory', 'connective tissue', 'dead', 'non-neoplastic epithelial']
PANNUKE_CLASSES = ['', 'neoplastic', 'inflammatory', 'connective tissue', 'dead', 'epithelial']

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5

def unnormalize_and_clamp_to_zero_to_one(img):
    return torch.clamp(unnormalize_to_zero_to_one(img.cpu()), 0, 1)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class ToTensorNoNorm():
    def __call__(self, X_i):
        X_i = np.array(X_i)
        if len(X_i.shape) == 2:
            # Add channel dim.
            X_i = X_i[:, :, None]
        return torch.from_numpy(np.array(X_i, copy=False)).permute(2, 0, 1)
    
def interpolate_3d(x, *args, **kwargs):
    return F.interpolate(x.unsqueeze(0), *args, **kwargs).squeeze(0)

class RandomResize(nn.Module):
    def __init__(self, scale=(0.5, 2.0), mode='nearest'):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def get_random_scale(self):
        return random.uniform(*self.scale)
    
    def forward(self, x):
        random_scale = self.get_random_scale()
        x = interpolate_3d(x, scale_factor=random_scale, mode=self.mode)
        return x
    
def read_jsonl(jsonl_path):
    import jsonlines
    lines = []
    with jsonlines.open(jsonl_path, 'r') as f:
        for line in f.iter():
            lines.append(line)
    return lines

class PannukeDataset(Dataset):
    def __init__(
        self,
        root="",
        split='train',
        side_x=128,
        side_y=128,
        shuffle=False,
        caption_list_dir='',
        augmentation_type='flip',
        exp_name=''
    ):
        super().__init__()
        self.root = Path(root)
        self.image_dir = os.path.join(self.root, 'classes', split)
        self.label_dir = os.path.join(self.root, 'classes', split)
        self.hv_dir = os.path.join(self.root, 'dist_masks', split)
        
        self.exp_name = exp_name
        if "+" in caption_list_dir:
            orig_caption_list_dir = caption_list_dir.split("+")[0]
            plus_setting = caption_list_dir.split("+")[1]

        json_fn = f'pannuke_{split}_{plus_setting}.json'

        captions_jsonl = read_jsonl(osp.join(orig_caption_list_dir, json_fn))
        self.caption_dict = {}
        for caption_jsonl in captions_jsonl:
            self.caption_dict[osp.splitext(caption_jsonl['file_name'])[0]] = caption_jsonl['text']
        
        self.split = split
        self.shuffle = shuffle
        self.side_x = side_x
        self.side_y = side_y

        if augmentation_type == 'none':
            self.augmentation = Compose([
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                # ToTensor(),
            ])
        elif augmentation_type == 'flip':
            self.augmentation = Compose([
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                RandomHorizontalFlip(p=0.5),
                # ToTensor(),
            ])
        elif 'resizedCrop' in augmentation_type:
            scale = [float(s) for s in augmentation_type.split('_')[1:]]
            assert len(scale) == 2, scale
            self.augmentation = Compose([
                RandomResize(scale=scale, mode='nearest'),
                RandomCrop((1024, 2048)),
                Resize((side_x, side_y), interpolation=InterpolationMode.NEAREST),
                RandomHorizontalFlip(p=0.5),
                # ToTensor(),
            ])
        elif 'pannuke' in augmentation_type:
            self.augmentation = Compose([
                # Resize((500, 500), interpolation=InterpolationMode.NEAREST),
                # RandomCrop((side_x, side_y)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                # ToTensor(),
            ])
        else:
            raise NotImplementedError(augmentation_type)
        
        # verification
        # self.images = sorted([osp.join(self.image_dir, file) for file in os.listdir(self.image_dir)
        #                       if osp.splitext(file)[0] in self.caption_dict.keys()])
        # self.labels = sorted([osp.join(self.label_dir, file) for file in os.listdir(self.label_dir)
        #                       if osp.splitext(file)[0] in self.caption_dict.keys()])
        # self.hvs    = sorted([osp.join(self.hv_dir, file) for file in os.listdir(self.hv_dir)
        #                       if osp.splitext(file)[0] in self.caption_dict.keys() and file.split(".")[-1].lower() in ["npy"]])

        self.images = sorted([osp.join(self.image_dir, file) for file in os.listdir(self.image_dir)
                              if "." in file and file.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif"]])
        self.labels = sorted([osp.join(self.label_dir, file) for file in os.listdir(self.label_dir)
                              if "." in file and file.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif"]])
        self.hvs = sorted([osp.join(self.hv_dir, file) for file in os.listdir(self.hv_dir)
                              if "." in file and file.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif", "npy"]])
        
        assert len(self.images) == len(self.labels), f'{len(self.images)} != {len(self.labels)}'
        assert len(self.images) == len(self.labels) == len(self.hvs), \
            f'{len(self.images)} != {len(self.labels)} != {len(self.hvs)}'
        for img, lbl, hv in zip(self.images, self.labels, self.hvs):
            assert osp.splitext(osp.basename(img))[0] == osp.splitext(osp.basename(lbl))[0]
            assert osp.splitext(osp.basename(img))[0] == osp.splitext(osp.basename(hv))[0]

    def __len__(self):
        return len(self.images)
    
    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))
    
    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)
    
    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    def get_caption_list_objects(self, idx):
        filename = osp.splitext(osp.basename(self.images[idx]))[0]
        filename = filename.split("sem_")[-1]
        caption = random.choice(self.caption_dict[filename])
        return caption
    
    def __getitem__(self, idx):
        try:
            original_pil_target = Image.open(self.labels[idx])
            original_npy_hv   = np.load(self.hvs[idx])
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {self.images[idx]}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        # Transforms
        # image = ToTensor()(original_pil_image)
        image = (ToTensorNoNorm()(original_pil_target) > 0).float()
        hv = ToTensorNoNorm()(original_npy_hv).float() # 2 channel
        image = torch.cat([image, hv])
        label = ToTensorNoNorm()(original_pil_target).float()
        img_lbl = self.augmentation(torch.cat([image, label]))

        caption = self.get_caption_list_objects(idx)
        cur_classes = np.unique(original_pil_target)

        if ("class" in self.exp_name):
            # shuffle classes and add to prompt
            if len(cur_classes)==1:
                caption += "; no nucleus"
            else:
                caption += "; including "
                included = []
                for i in cur_classes[1:]:
                    included.append(PANNUKE_CLASSES[i])
                random.shuffle(included)

                if len(included) > 1:
                    included[-1] = "and " + included[-1]
                    if len(included)==2:
                        included = ' '.join(included)
                    else:
                        included = ', '.join(included)
                else:
                    included = included[0]
                caption += included
        
        return img_lbl[:3], img_lbl[3:], caption # img, lbl, caption
    
def transform_lbl(lbl: torch.Tensor, *args, **kwargs):
    lbl = lbl.long()
    if lbl.size(1) == 1:
        # Remove single channel axis.
        lbl = lbl[:, 0]
    rgbs = colors[lbl]
    rgbs = rgbs.permute(0, 3, 1, 2)
    return rgbs / 255.
