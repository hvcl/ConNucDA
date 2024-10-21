import json
import os
import cv2
import numpy as np
import random

from torch.utils.data import Dataset
from PIL import Image

# "images": "images/train/consep_10__0_0.png", 
# "classes": "classes/train/consep_10__0_0.png", 
# "instances": "instances/train/consep_10__0_0.png", 
# "prompt": "High quality histopathology colon tissue image including nuclei types of epithelial, lymphocyte, and connective tissue"


##### Errored #####
# PANNUKE_CLASSES = [
#     '',
#     'neoplastic', 
#     'inflammatory', 
#     'connective tissue', 
#     'dead', 
#     'non-neoplastic epithelial'
# ]
###################

PANNUKE_CLASSES = [
    '', 
    'inflammatory', 
    'connective tissue', 
    'dead', 
    'non-neoplastic epithelial', 
    'neoplastic'
]

# inflammatory	connective	dead	epithelial	neoplastic

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./setup_pathology_ds/v3/pannuke_test.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.pannuke_data_path = '/Dataset/pannuke/splits/'
        self.num_classes = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_fn = item['images']
        cls_fn = item['classes']
        ist_fn = item['instances']
        fn_ext = ist_fn.split('test/')[-1]

        fn_npy = os.path.basename(cls_fn).split('.png')[0] + '.npy'
        hv_fn = os.path.join(self.pannuke_data_path, 'dist_masks/test', fn_npy)
        tissue_type = fn_ext.split("inst_")[-1].split("_f")[0].lower()

        if tissue_type == "adrenal_gland":
            tissue_type = "adrenal gland"
        elif tissue_type == "bile_duct":
            tissue_type == "bile duct"
        elif tissue_type == "headneck":
            tissue_type == "head and neck"

        # if "-" in tissue_type:
        #     str1, str2 = tissue_type.split("-")
        #     tissue_type = " ".join([str1, str2])
        # elif "_" in tissue_type:
        #     str1, str2 = tissue_type.split("_")
        #     tissue_type = " ".join([str1, str2])

        img = cv2.imread(self.pannuke_data_path + img_fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hv = np.load(hv_fn).astype(np.float32)

        # Normalize original images to [0, 1].
        orig = (img.astype(np.float32) / 255.)
        img = (img.astype(np.float32) / 127.5) - 1.0

        cls = np.array(Image.open(self.pannuke_data_path + cls_fn), dtype=np.uint8)
        ist = np.array(Image.open(self.pannuke_data_path + ist_fn), dtype=np.int16)

        cls = cls.astype(np.uint8)
        ist = ist.astype(np.int16)

        # set the prompt
        prompt = self.set_prompt(tissue_type, cls)

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
    
    def set_prompt(self, tissue_type, cls):
        prompt = f"high quality histopathology {tissue_type} tissue nuclei image"

        cls_now = np.unique(cls)

        # shuffle classes and add to prompt
        if len(cls_now)==1:
            prompt += " without nucleus"
        else:
            prompt += " including nuclei types of "
            included = []
            for i in cls_now[1:]:
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
            prompt += included

        return prompt