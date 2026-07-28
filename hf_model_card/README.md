---
license: other
tags:
  - pathology
  - nuclei
  - diffusion
  - label-generation
library_name: pytorch
---

# ConNucDA model release

This repository hosts model weights associated with **ConNucDA: Controllable and Efficient Multi-Class Pathology Nuclei Data Augmentation using Text-Conditioned Diffusion Models** (MICCAI 2024).

## Update: 2026-07-28

This release adds the recovered Lizard label-generation checkpoint:

| File | Component | Dataset | Resolution | Training step |
|---|---|---|---:|---:|
| `label_synthesis/lizard_gcdp_base_256x256_checkpoint.300000.pt` | GCDP label generator | Lizard | 256 x 256 | 300,000 |

The file is a label/layout generator used in the data-generation pipeline. It is **not** a PathLDM/ControlNet image-synthesis checkpoint.

## Image-synthesis initialization and checkpoint availability

The ConNucDA image-synthesis model was fine-tuned from the PathLDM ControlNet initialization
`control_plip_imagenet_ini_<dataset>_seg_edge_hv.ckpt`. The original final image-synthesis checkpoints for Lizard, PanNuke, and EndoNuke are not available in the release workspace at the time of this update. They are therefore not substituted with another model in this repository; they will be added once recovered and verified.

The accompanying source code records the image-synthesis checkpoints used by the sampling scripts and the corresponding training entry points:

- `image_synthesis/train-seg_edge_hv.py`
- `image_synthesis/train-seg_edge_hv-resume.py`
- `image_synthesis/sample_img_synthetic_{lizard,pannuke,endonuke}.py`

## Loading the released label model

Use the GCDP label-synthesis implementation distributed in the ConNucDA source repository. Pass this file through its `--checkpoint_path` option, with the Lizard 256 x 256 configuration in `label_synthesis/`.

## License and intended use

The checkpoint is released for non-commercial research and reproducibility. Users must comply with the terms of the underlying datasets and the respective upstream software licenses.

## Citation

```bibtex
@inproceedings{oh2024connucda,
  title={Controllable and Efficient Multi-Class Pathology Nuclei Data Augmentation using Text-Conditioned Diffusion Models},
  author={Oh, Hyun-Jic and Jeong, Won-Ki},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
  pages={36--46},
  year={2024},
  organization={Springer}
}
```
