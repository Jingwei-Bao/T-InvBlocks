<div align="center">

# Tri-Branch Invertible Block for Image Rescaling (T-InvBlock)

</div>

<div align="center">

[![AAAI](https://img.shields.io/badge/AAAI%202025-Accepted-informational.svg)](https://openreview.net/forum?id=gTQ1vb4wZj&noteId=EjBWHdePwl)
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.12508-b31b1b.svg)](https://arxiv.org/abs/2412.13508)
![Visitor](https://visitor-badge.laobi.icu/badge?page_id=Jingwei-Bao/T-InvBlocks)

[Jingwei Bao](https://scholar.google.com.hk/citations?user=9rInCsUAAAAJ), [Jinhua Hao](https://eric-hao.github.io/), [Pengcheng Xu](), [Ming Sun](https://msunming.github.io/), [Chao Zhou](), [Shuyuan Zhu]()

</div>

> This repository is the official implementation of the paper [Plug-and-Play Tri-Branch Invertible Block for Image Rescaling](https://arxiv.org/abs/2412.13508) (AAAI 2025).

## 💥 News

* **2024-12:** The [paper](https://arxiv.org/abs/2412.13508) and the corresponding [code](https://github.com/Jingwei-Bao/T-InvBlocks) are released.


## 🛠️ Dependencies and Installation
The codes are developed under the following environments:
```shell
# 1. Python 3.7.1 (Recommend to use conda)
conda create -n tinvb python=3.7.1
conda activate tinvb

# 2. PyTorch=1.9.0, torchvision=0.10.0, cudatoolkit=11.1
python -m pip install --upgrade pip
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Other dependencies
pip install -r requirements.txt
```

## 📚 Dataset Preparation
We use the DIV2K training split for model training, and validate on DIV2K validation split and four widely-used benchmarks: Set5, Set14, BSDS100, and Urban100. 

Please organize the datasets and the code in the following folder structure:
<details>
<summary><b>Folder Structure for Datasets</b></summary>

```
├── datasets
│   ├── BSDS100
│   │   └── *.png
│   ├── DIV2K
│   │   ├── DIV2K_train_HR
│   │   │   └── *.png
│   │   ├── DIV2K_train_LR_bicubic
│   │   │   ├── X2
│   │   │   │   └── *.png
│   │   │   └── X4
│   │   │       └── *.png
│   │   ├── DIV2K_valid_HR
│   │   │   └── *.png
│   │   └── DIV2K_valid_LR_bicubic
│   │       ├── X2
│   │       │   └── *.png
│   │       └── X4
│   │           └── *.png
│   ├── Set5
│   │   ├── GTmod12
│   │   │   └── *.png
│   │   ├── LRbicx2
│   │   │   └── *.png
│   │   └── LRbicx4
│   │       └── *.png
│   ├── Set14
│   │   ├── GTmod12
│   │   │   └── *.png
│   │   ├── LRbicx2
│   │   │   └── *.png
│   │   └── LRbicx4
│   │       └── *.png
│   └── urban100
│       └── *.png
└── TInvBlock 
    ├── codes
    ├── experiments
    ├── results
    └── tb_logger
```

To accelerate training, we suggest [crop the 2K resolution images to sub-images](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md#div2k) for faster IO speed.

</details>



## 🎯 Testing 
The pretrained models are available in `./experiments/pretrained_TIRN` and `./experiments/pretrained_TSAIN`. The test config files are located in `./codes/options/test` for quickly reproducing the results reported in the paper.

T-IRN:
```shell
# For scale x2, change directory to `.code/`, run
python test.py -opt options/test/TIRN_2.yml 

# For scale x4, change directory to `.code/`, run
python test.py -opt options/test/TIRN_4.yml
```

T-SAIN:
```shell
# For scale x2 with JPEG compression QF=90, change directory to `.code/`, run
python test.py -opt options/test/TSAIN_2.yml -format JPEG -qf 90

# For scale x4 with JPEG compression QF=90, change directory to `.code/`, run
python test.py -opt options/test/TSAIN_4.yml -format JPEG -qf 90
```
<details>
<summary><b>State of the Art Performance</b></summary>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-set5-2x)](https://paperswithcode.com/sota/image-rescaling-on-set5-2x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-set14-2x)](https://paperswithcode.com/sota/image-rescaling-on-set14-2x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-bsd100-2x)](https://paperswithcode.com/sota/image-rescaling-on-bsd100-2x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-urban100-2x)](https://paperswithcode.com/sota/image-rescaling-on-urban100-2x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-div2k-val-2x)](https://paperswithcode.com/sota/image-rescaling-on-div2k-val-2x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-div2k-val-4x)](https://paperswithcode.com/sota/image-rescaling-on-div2k-val-4x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-div2k-val-q30-2x)](https://paperswithcode.com/sota/image-rescaling-on-div2k-val-q30-2x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-div2k-val-q50-2x)](https://paperswithcode.com/sota/image-rescaling-on-div2k-val-q50-2x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-div2k-val-q70-2x)](https://paperswithcode.com/sota/image-rescaling-on-div2k-val-q70-2x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-div2k-val-q30-4x)](https://paperswithcode.com/sota/image-rescaling-on-div2k-val-q30-4x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-div2k-val-q50-4x)](https://paperswithcode.com/sota/image-rescaling-on-div2k-val-q50-4x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/plug-and-play-tri-branch-invertible-block-for/image-rescaling-on-div2k-val-q70-4x)](https://paperswithcode.com/sota/image-rescaling-on-div2k-val-q70-4x?p=plug-and-play-tri-branch-invertible-block-for)&nbsp;

</details>

## 🚀 Training
The training configs are included in `./codes/options/train`. 
 
T-IRN:
```shell
# For scale x2, change directory to `.code/`, run
python train.py -opt options/train/TIRN_2.yml 

# For scale x4, change directory to `.code/`, run
python train.py -opt options/train/TIRN_4.yml
```

T-SAIN:
```shell
# For scale x2 with JPEG compression QF=90, change directory to `.code/`, run
python train.py -opt options/train/TSAIN_2.yml 

# For scale x4 with JPEG compression QF=90, change directory to `.code/`, run
python train.py -opt options/train/TSAIN_4.yml 
```

## 🙌🏻️ Acknowledgement
The code is based on [SAIN](https://github.com/yang-jin-hai/SAIN), [IRN](https://github.com/pkuxmq/Invertible-Image-Rescaling/tree/ECCV) and [BasicSR](https://github.com/xinntao/BasicSR).

## 🔍 Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@inproceedings{bao2025plug,
  title={Plug-and-play tri-branch invertible block for image rescaling},
  author={Bao, Jingwei and Hao, Jinhua and Xu, Pengcheng and Sun, Ming and Zhou, Chao and Zhu, Shuyuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={2},
  pages={1826--1834},
  year={2025}
}
