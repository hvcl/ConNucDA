import json
import glob
import os
import numpy as np
from PIL import Image

# V1
# V2: colon -> endometrium

base_prompt = "high quality histopathology IHC-stained endometrium tissue nuclei image"

data_pth = '/Dataset/endonuke/splits'
target = "val"

ist_pths = sorted(glob.glob(data_pth + f'/instances/{target}/*.png'))
cls_pths = sorted(glob.glob(data_pth + f'/classes/{target}/*.png'))
pnt_pths = sorted(glob.glob(data_pth + f'/points_v2/{target}/*.png'))
img_pths = sorted(glob.glob(data_pth + f'/images/{target}/*.png'))

assert len(ist_pths) == len(cls_pths) == len(img_pths) == len(pnt_pths)

classes = ['', 'stroma', 'epithelium', 'others']

# with open(json_file_path, 'w') as json_file:
#     json.dump(line, json_file)

data_dicts = []
# for idx, (dst_pth, ist_pth, cls_pth, img_pth) in enumerate(zip(dst_pths, ist_pths, cls_pths, img_pths)):
for idx, (ist_pth, cls_pth, img_pth, pnt_pth) in enumerate(zip(ist_pths, cls_pths, img_pths, pnt_pths)):
    fn_ext = os.path.basename(ist_pth)

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

    line = {"images": f"images/{target}/{fn_ext}", 
            "classes": f"classes/{target}/{fn_ext}", 
            "instances": f"instances/{target}/{fn_ext}", 
            "points": f"points/{target}/{fn_ext}", 
            "prompt": prompt}
    data_dicts.append(line)

json_file_path = f'endonuke_{target}.json'
with open(json_file_path, 'w') as json_file:
    for data_dict in data_dicts:
        json.dump(data_dict, json_file)
        json_file.write('\n')
