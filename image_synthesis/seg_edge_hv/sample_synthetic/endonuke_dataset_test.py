import os
import cv2
import json
import glob
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

# "images": "images/train/consep_10__0_0.png", 
# "classes": "classes/train/consep_10__0_0.png", 
# "instances": "instances/train/consep_10__0_0.png", 
# "prompt": "High quality histopathology endometrium tissue image including nuclei types of epithelial, lymphocyte, and connective tissue"


ENDONUKE_CLASSES = ['', 'stroma', 'epithelium', 'others']

class MyDataset(Dataset):
    def __init__(self, syn_set):
        # self.data = []
        # with open('./setup_pathology_ds/synthetic/endonuke_test.json', 'rt') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))

        self.endonuke_data_path = f'/Dataset/endonuke/splits/MICCAI2024_synthetic/{syn_set}'
        self.img_dir = f'/Dataset/endonuke/splits/images/test' if 'test' in syn_set \
                    else f'/Dataset/endonuke/splits/images/val'
        self.img_pths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))[120:]
        self.cls_pths = sorted(glob.glob(os.path.join(self.endonuke_data_path, "classes", "*.png")))[120:]
        self.ist_pths = sorted(glob.glob(os.path.join(self.endonuke_data_path, "instances", "*.png")))[120:]
        self.dst_pths = sorted(glob.glob(os.path.join(self.endonuke_data_path, "dist_masks", "*.npy")))[120:]
        self.num_classes = 4

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, idx):
        # item = self.data[idx]
        
        img_fn = self.img_pths[idx]
        cls_fn = self.cls_pths[idx]
        ist_fn = self.ist_pths[idx]
        fn_npy = self.dst_pths[idx]
        # cls_fn = item['classes']
        # ist_fn = item['instances']
        
        fn_ext = ist_fn.split('instances/')[-1]
        # fn_ext = ist_fn.split('test/')[-1]
        
        # fn_npy = os.path.basename(ist_fn).split('.png')[0] + '.npy'
        # hv_fn = os.path.join(self.lizard_data_path, 'dist_masks/test', fn_npy)
        # hv_fn = os.path.join(self.lizard_data_path, 'dist_masks', fn_npy)

        # img = cv2.imread(self.lizard_data_path + img_fn)
        # img = cv2.imread(self.img_dir + img_fn)
        img = cv2.imread(img_fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # hv = np.load(hv_fn).astype(np.float32)
        hv = np.load(fn_npy).astype(np.float32)

        # Normalize original images to [0, 1].
        orig = (img.astype(np.float32) / 255.)
        img = (img.astype(np.float32) / 127.5) - 1.0

        cls = np.array(Image.open(cls_fn), dtype=np.uint8)
        ist = np.array(Image.open(ist_fn), dtype=np.int16)

        cls = cls.astype(np.uint8)
        ist = ist.astype(np.int16)

        # set the prompt
        prompt = self.set_prompt(cls)

        # Preprocess semantic and instance maps
        cond = self.preprocess_input(cls, ist, hv) # [H,W,C]

        return dict(orig=orig, img=img, txt=prompt, hint=cond, fn_ext=fn_ext)

    def preprocess_input(self, cls, ist, hv):
        # data['label'] is already a numpy array, so we don't need to change its type
        # However, if you want to ensure it's an integer type, you can uncomment the next line
        # data['label'] = data['label'].astype(np.int64)

        # create one-hot label map
        # label_map = np.expand_dims(cls, axis=-1)
        label_map = cls
        # bs, h, w = label_map.shape
        nc = self.num_classes

        # Create one-hot encoding
        onehot_semantic_map = (np.arange(nc) == label_map[..., None]).astype(np.float32) # [B,C,H,W,NC]

        # Expand channel of instance map
        # instance_map = np.expand_dims(ist) #, axis=-1)
        edge_map = self.get_edges(ist)
        edge_map = np.expand_dims(edge_map, axis=-1)

        # instance map
        input_semantics = np.concatenate((onehot_semantic_map, edge_map, hv), axis=-1)

        return input_semantics

    def get_edges(self, t):
        edge = np.zeros_like(t, dtype=np.uint8)
        # edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        # edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        # edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        # edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, 1:]  = edge[:, 1:]  | (t[:, 1:] != t[:, :-1])
        edge[:, :-1] = edge[:, :-1] | (t[:, 1:] != t[:, :-1])
        edge[1:, :]  = edge[1:, :]  | (t[1:, :] != t[:-1, :])
        edge[:-1, :] = edge[:-1, :] | (t[1:, :] != t[:-1, :])
        return edge.astype(np.float32)
    
    def set_prompt(self, cls): #, tissue_type):
        prompt = f"high quality histopathology IHC-stained endometrium tissue nuclei image"

        cls_now = np.unique(cls)

        # shuffle classes and add to prompt
        if len(cls_now)==1:
            prompt += " without nucleus"
        else:
            prompt += " including nuclei types of "
            included = []
            for i in cls_now[1:]:
                included.append(ENDONUKE_CLASSES[i])
            random.shuffle(included)

            if len(included) > 1:
                included[-1] = "and " + included[-1]
                if len(included)==2:
                    included = ' '.join(included)
                else:
                    included = ', '.join(included)
            else:
                included = included[0]
            prompt += included

        return prompt