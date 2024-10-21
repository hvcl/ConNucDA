# Step 1 - Get a dataset

# Step 2 - Load the dataset
edit tutorial_dataset.py

# Step 3 - What SD model do you want to control?
## seg + edge
python tool_add_control.py ./pathldm/plip_imagenet_finetune/plip_imagenet_finetune/checkpoints/epoch_3.ckpt ./pathldm/control/control_plip_imagenet_ini_lizard.ckpt ./models/cldm_pathLDM_lizard.yaml
python tool_add_control.py ./pathldm/plip_imagenet_finetune/plip_imagenet_finetune/checkpoints/epoch_3.ckpt ./pathldm/control/control_plip_imagenet_ini_pannuke.ckpt ./models/cldm_pathLDM_pannuke.yaml
python tool_add_control.py ./pathldm/plip_imagenet_finetune/plip_imagenet_finetune/checkpoints/epoch_3.ckpt ./pathldm/control/control_plip_imagenet_ini_endonuke.ckpt ./models/cldm_pathLDM_endonuke.yaml

## seg + edge + hv
python tool_add_control.py ./pathldm/plip_imagenet_finetune/plip_imagenet_finetune/checkpoints/epoch_3.ckpt ./pathldm/control/control_plip_imagenet_ini_lizard_seg_edge_hv.ckpt ./models/cond_seg_edge_hv/cldm_pathLDM_lizard.yaml
python tool_add_control.py ./pathldm/plip_imagenet_finetune/plip_imagenet_finetune/checkpoints/epoch_3.ckpt ./pathldm/control/control_plip_imagenet_ini_pannuke_seg_edge_hv.ckpt ./models/cond_seg_edge_hv/cldm_pathLDM_pannuke.yaml
python tool_add_control.py ./pathldm/plip_imagenet_finetune/plip_imagenet_finetune/checkpoints/epoch_3.ckpt ./pathldm/control/control_plip_imagenet_ini_endonuke_seg_edge_hv.ckpt ./models/cond_seg_edge_hv/cldm_pathLDM_endonuke.yaml

## 3-2. test on stable diffusion
<!-- python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt ./models/cldm_v15.yaml
python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt ./models/cldm_v15.yaml -->


#### start setting environment
```
conda env create -f environment.yaml
conda activate control
```

#### text prompt
##### 24-02-08
Tissue type / Ratio (Proportion) / Class
lizard - v4 (As v1~v3 has error: no eosinophil which is a 5th class originally.)
pannuke - v1
##### 24-02-11
Staining (H&E, IHC)
lizard - v5
pannuke - v2
endonuke
##### 24-02-22
pannuke - v3
