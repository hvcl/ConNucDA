# ConNucDA: Controllable Multi-Class Pathology Nuclei Data Augmentation
Pytorch implementation of Controllable and Efficient Multi-Class Pathology Nuclei Data Augmentation using Text-Conditioned Diffusion Models (MICCAI 2024)

This repository contains the official implementation of the paper:

**"Controllable and Efficient Multi-Class Pathology Nuclei Data Augmentation using Text-Conditioned Diffusion Models" (MICCAI2024)** 

- MICCAI version: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_4)
- Arvix version: [Paper](https://arxiv.org/abs/2407.14434)

## Updates

### 2026-07-28

- Released the recovered Lizard GCDP label-generation checkpoint and model card on [Hugging Face](https://huggingface.co/Hyun-Jic/ConNucDA).
- Documented that the image-synthesis experiments fine-tuned the PathLDM ControlNet initialization (`control_plip_imagenet_ini_<dataset>_seg_edge_hv.ckpt`).
- Documented the training entry points and final checkpoint references used by the released sampling scripts.

> **Checkpoint scope:** the Hugging Face release currently contains the Lizard GCDP label generator only. It is not an image-synthesis ControlNet checkpoint. The original final image-synthesis weights are not present in the release workspace and will be added once recovered and verified.

## Overview

We present a novel approach for multi-class pathology nuclei data augmentation using text-conditioned diffusion models. Our method offers controllable and efficient synthesis of both nuclei labels and images, addressing the challenges of limited and imbalanced datasets in pathology image analysis.

## Repository Structure

The repository is organized into two main directories:

- `label_synthesis/`: Code for generating synthetic nuclei labels using text-conditioned diffusion models
- `image_synthesis/`: Code for synthesizing pathology images based on ControlNet

### Features

- Text-guided control over nuclei characteristics (e.g., size, shape, density)
- Multi-class label generation
- Efficient synthesis process

### Installation

```bash
cd label_synthesis
pip install -r requirements.txt
```

### Checkpoints

- Label synthesis: the recovered Lizard GCDP 256 x 256 checkpoint (300,000 training steps) is available on [Hugging Face](https://huggingface.co/Hyun-Jic/ConNucDA).
- Image synthesis: the model was initialized from PathLDM ControlNet weights and then fine-tuned with `image_synthesis/train-seg_edge_hv.py` or `image_synthesis/train-seg_edge_hv-resume.py`. The released sampling scripts reference the final runs as Lizard `epoch=53-step=20735.ckpt`, PanNuke `epoch=249-step=49999.ckpt`, and EndoNuke `epoch=1041-step=49999.ckpt`.
- The original final image-synthesis files themselves are not present locally, so they are not substituted with different weights in the Hugging Face release.

### To-do-list
- Data link and preprocessing codes

### Acknowledgements

We would like to acknowledge the following projects that have contributed to our work:

- Label synthesis part of this project is built upon the work of [GCDP](https://github.com/pmh9960/GCDP). We thank the authors for making their code available.
- Image synthesis part of our project utilizes [ControlNet](https://github.com/lllyasviel/ControlNet) as a baseline. We are grateful to the ControlNet team for their excellent work and open-source contribution.

We express our sincere gratitude to the authors and contributors of these projects for their valuable work which has significantly aided our research.

### Citation

If you find this work useful in your research, please consider citing our paper:
```bibtex
@InProceedings{10.1007/978-3-031-72083-3_4,
author="Oh, Hyun-Jic
and Jeong, Won-Ki",
editor="Linguraru, Marius George
and Dou, Qi
and Feragen, Aasa
and Giannarou, Stamatia
and Glocker, Ben
and Lekadir, Karim
and Schnabel, Julia A.",
title="Controllable and Efficient Multi-class Pathology Nuclei Data Augmentation Using Text-Conditioned Diffusion Models",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="36--46",
abstract="In the field of computational pathology, deep learning algorithms have made significant progress in tasks such as nuclei segmentation and classification. However, the potential of these advanced methods is limited by the lack of available labeled data. Although image synthesis via recent generative models has been actively explored to address this challenge, existing works have barely addressed label augmentation and are mostly limited to single-class and unconditional label generation. In this paper, we introduce a novel two-stage framework for multi-class nuclei data augmentation using text-conditional diffusion models. In the first stage, we innovate nuclei label synthesis by generating multi-class semantic labels and corresponding instance maps through a joint diffusion model conditioned by text prompts that specify the label structure information. In the second stage, we utilize a semantic and text-conditional latent diffusion model to efficiently generate high-quality pathology images that align with the generated nuclei label images. We demonstrate the effectiveness of our method on large and diverse pathology nuclei datasets, with evaluations including qualitative and quantitative analyses, as well as assessments of downstream tasks.",
isbn="978-3-031-72083-3"
}
```
