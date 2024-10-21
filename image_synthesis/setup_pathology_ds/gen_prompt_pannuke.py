import json
import glob
import os
import numpy as np
from PIL import Image

# V1
# V2: "hematoxylin and eosin stained" added
# V3: class sequence 수정, tissue type headneck->head and neck

data_pth = '/Dataset/pannuke/splits'
target = "val" # train val test

ist_pths = sorted(glob.glob(data_pth + f'/instances/{target}/*.png'))
cls_pths = sorted(glob.glob(data_pth + f'/classes/{target}/*.png'))
img_pths = sorted(glob.glob(data_pth + f'/images/{target}/*.png'))

assert len(ist_pths) == len(cls_pths) == len(img_pths)

# classes = ['', 'neoplastic', 'inflammatory', 'connective tissue', 'dead', 'non-neoplastic epithelial']
classes = ['', 'inflammatory', 'connective tissue', 'dead', 'non-neoplastic epithelial', 'neoplastic']
# PANNUKE_CLASSES = ['', 'inflammatory', 'connective tissue', 'dead', 'epithelial', 'non-neoplastic epithelial']

data_dicts = []
# for idx, (dst_pth, ist_pth, cls_pth, img_pth) in enumerate(zip(dst_pths, ist_pths, cls_pths, img_pths)):
for idx, (ist_pth, cls_pth, img_pth) in enumerate(zip(ist_pths, cls_pths, img_pths)):
    fn_ist_ext = os.path.basename(ist_pth)
    fn_cls_ext = os.path.basename(cls_pth)
    fn_img_ext = os.path.basename(img_pth)

    tissue_type = fn_ist_ext.split("inst_")[-1].split("_f")[0].lower()
    if tissue_type == "adrenal_gland":
        tissue_type = "adrenal gland"
    elif tissue_type == "bile_duct":
        tissue_type == "bile duct"
    elif tissue_type == "headneck":
        tissue_type == "head and neck"
    # if "_" in tissue_type:
    #     str1, str2 = tissue_type.split("_")
    #     tissue_type = str1 + " " + str2
    # elif "-" in tissue_type:
    #     str1, str2 = tissue_type.split("-")
    #     tissue_type = str1 + " " + str2
    
    base_prompt = f"high quality H&E-stained histopathology {tissue_type} tissue nuclei image"

    cls = np.array(Image.open(cls_pth), dtype=np.uint8)
    prompt = f"{base_prompt} including nuclei types of "

    included = []
    for i in np.unique(cls)[1:]:
        included.append(classes[i])
    
    # random.shuffle(included)

    if len(included) > 1:
        included[-1] = "and " + included[-1]
        if len(included)==2:
            included = ' '.join(included)
        else:
            included = ', '.join(included)
    elif len(included) == 1:
        included = included[0]
    else:
        prompt = f"{base_prompt} without nuclei"
        included = ""

    prompt += included
    # print(prompt)
    # exit()

    line = {"images": f"images/{target}/{fn_img_ext}", 
            "classes": f"classes/{target}/{fn_cls_ext}", 
            "instances": f"instances/{target}/{fn_ist_ext}", 
            # "points": f"points/{target}/{fn_ist_ext}", 
            "prompt": prompt}
    data_dicts.append(line)

json_file_path = f'pannuke_{target}.json'
with open(json_file_path, 'w') as json_file:
    for data_dict in data_dicts:
        json.dump(data_dict, json_file)
        json_file.write('\n')
