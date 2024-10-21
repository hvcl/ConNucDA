import json
import cv2
import numpy as np
import random

from torch.utils.data import Dataset
from PIL import Image
# "images": "images/train/consep_10__0_0.png", 
# "classes": "classes/train/consep_10__0_0.png", 
# "instances": "instances/train/consep_10__0_0.png", 
# "prompt": "A pathology image including nuclei types of epithelial, lymphocyte, and connective tissue"

nuclei_types = [
    'Neutrophil',
    'Epithelial',
    'Lymphocyte',
    'Plasma',
    'Neutrophil',
    'Connective_tissue'
]

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./setup_pathology_ds/v3/lizard_train.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.lizard_data_path = '/Dataset/lizard/NASDM/lizard_split_norm_256/'
        self.num_classes = 7

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # cls_fn = item['classes']
        # ist_fn = item['instances']
        img_fn = item['images']
        pnt_fn = item['points']
        # if ('including' in item['prompt']) and ('and' in item['prompt']):
        #     # shuffle nuclei text sequence
        #     nuclei_types_in_prompt = []
        #     for nuclei_type in nuclei_types:
        #         if nuclei_type in item['prompt']:
        #             nuclei_types_in_prompt.append(nuclei_type)
        #     random.shuffle(nuclei_types_in_prompt)
        
        #     ## V1
        #     # # "A pathology image including nuclei types of epithelial, lymphocyte, and connective tissue"
        #     # if len(nuclei_types_in_prompt)==2:
        #     #     item['prompt'] = f"A pathology image including nuclei types of {nuclei_types_in_prompt[0]} and {nuclei_types_in_prompt[1]}"
        #     # elif len(nuclei_types_in_prompt)==3:
        #     #     item['prompt'] = f"A pathology image including nuclei types of {nuclei_types_in_prompt[0]}, {nuclei_types_in_prompt[1]}, and {nuclei_types_in_prompt[2]}"
        #     # elif len(nuclei_types_in_prompt)==4:
        #     #     item['prompt'] = f"A pathology image including nuclei types of {nuclei_types_in_prompt[0]}, {nuclei_types_in_prompt[1]}, {nuclei_types_in_prompt[2]}, and {nuclei_types_in_prompt[3]}"
        #     # elif len(nuclei_types_in_prompt)==5:
        #     #     item['prompt'] = f"A pathology image including nuclei types of {nuclei_types_in_prompt[0]}, {nuclei_types_in_prompt[1]}, {nuclei_types_in_prompt[2]}, {nuclei_types_in_prompt[3]}, and {nuclei_types_in_prompt[4]}"
        #     # elif len(nuclei_types_in_prompt)==6:
        #     #     item['prompt'] = f"A pathology image including nuclei types of {nuclei_types_in_prompt[0]}, {nuclei_types_in_prompt[1]}, {nuclei_types_in_prompt[2]}, {nuclei_types_in_prompt[3]}, {nuclei_types_in_prompt[4]}, and {nuclei_types_in_prompt[5]}"

        #     ## V2
        #     # "A histopathology colon tissue nuclei image including nuclei types of epithelial, lymphocyte, and connective tissue"
        #     if len(nuclei_types_in_prompt)==2:
        #         item['prompt'] = f"A histopathology colon tissue nuclei image including nuclei types of {nuclei_types_in_prompt[0]} and {nuclei_types_in_prompt[1]}"
        #     elif len(nuclei_types_in_prompt)==3:
        #         item['prompt'] = f"A histopathology colon tissue nuclei image including nuclei types of {nuclei_types_in_prompt[0]}, {nuclei_types_in_prompt[1]}, and {nuclei_types_in_prompt[2]}"
        #     elif len(nuclei_types_in_prompt)==4:
        #         item['prompt'] = f"A histopathology colon tissue nuclei image including nuclei types of {nuclei_types_in_prompt[0]}, {nuclei_types_in_prompt[1]}, {nuclei_types_in_prompt[2]}, and {nuclei_types_in_prompt[3]}"
        #     elif len(nuclei_types_in_prompt)==5:
        #         item['prompt'] = f"A histopathology colon tissue nuclei image including nuclei types of {nuclei_types_in_prompt[0]}, {nuclei_types_in_prompt[1]}, {nuclei_types_in_prompt[2]}, {nuclei_types_in_prompt[3]}, and {nuclei_types_in_prompt[4]}"
        #     elif len(nuclei_types_in_prompt)==6:
        #         item['prompt'] = f"A histopathology colon tissue nuclei image including nuclei types of {nuclei_types_in_prompt[0]}, {nuclei_types_in_prompt[1]}, {nuclei_types_in_prompt[2]}, {nuclei_types_in_prompt[3]}, {nuclei_types_in_prompt[4]}, and {nuclei_types_in_prompt[5]}"

        ## TEMP ##
        item['prompt'] = "A histopathology colon tissue nuclei image"

        prompt = item['prompt']

        # cls = cv2.imread(self.lizard_data_path + cls_fn)
        # ist = cv2.imread(self.lizard_data_path + ist_fn)
        img = cv2.imread(self.lizard_data_path + img_fn)

        # Do not forget that OpenCV read images in BGR order.
        # cls = cv2.cvtColor(cls, cv2.COLOR_BGR2RGB)
        # ist = cv2.cvtColor(ist, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cls = np.array(Image.open(self.lizard_data_path + cls_fn), dtype=np.uint8)
        # ist = np.array(Image.open(self.lizard_data_path + ist_fn), dtype=np.int16)
        pnt = np.array(Image.open(self.lizard_data_path + pnt_fn), dtype=np.int16)
        
        # Normalize source images to [0, 1].
        # cls = cls.astype(np.uint8)
        # ist = ist.astype(np.int16)
        pnt = pnt.astype(np.uint8)

        # Preprocess semantic and instance maps
        # cond = self.preprocess_input(cls, ist) # [H,W,C]
        pnt = self.preprocess_input(pnt) # [H,W,C]

        # Normalize target images to [-1, 1].
        img = (img.astype(np.float32) / 127.5) - 1.0

        img = np.copy(img)
        # cond = np.copy(cond)
        pnt = np.copy(pnt)

        # Augmentation (random horizontal/vertical flip)
        if np.random.rand() > 0.5: # horizontal
            img = np.flip(img, axis=0)
            # cond = np.flip(cond, axis=0)
            pnt = np.flip(pnt, axis=0)
        if np.random.rand() > 0.5: # vertical
            img = np.flip(img, axis=1)
            # cond = np.flip(cond, axis=1)
            pnt = np.flip(pnt, axis=1)

        img = np.copy(img)
        pnt = np.copy(pnt)
        
        return dict(jpg=img, txt=prompt, hint=pnt)

    def preprocess_input(self, pnt):
        label_map = pnt
        # bs, h, w = label_map.shape
        nc = self.num_classes

        # Create one-hot encoding
        input_semantics = (np.arange(nc) == label_map[..., None]).astype(np.float32) # [B,C,H,W,NC]

        return input_semantics
    
    # def preprocess_input(self, cls, ist):
    #     # data['label'] is already a numpy array, so we don't need to change its type
    #     # However, if you want to ensure it's an integer type, you can uncomment the next line
    #     # data['label'] = data['label'].astype(np.int64)

    #     # create one-hot label map
    #     # label_map = np.expand_dims(cls, axis=-1)
    #     label_map = cls
    #     # bs, h, w = label_map.shape
    #     nc = self.num_classes

    #     # Create one-hot encoding
    #     onehot_semantic_map = (np.arange(nc) == label_map[..., None]).astype(np.float32) # [B,C,H,W,NC]

    #     # Expand channel of instance map
    #     # instance_map = np.expand_dims(ist) #, axis=-1)
    #     edge_map = self.get_edges(ist)
    #     edge_map = np.expand_dims(edge_map, axis=-1)

    #     # instance map
    #     input_semantics = np.concatenate((onehot_semantic_map, edge_map), axis=-1)

    #     return input_semantics

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