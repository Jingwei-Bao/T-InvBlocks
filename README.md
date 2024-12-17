# Plug-and-Play-Tri-Branch-Invertible-Block-for-Image-Rescaling

## Dependencies and Installation
The codes are developed under the following environments:
1. Python 3.7.1 (Recommend to use Anaconda)

```shell
conda create -n sain python=3.7.1
conda activate sain
```

2. PyTorch=1.9.0, torchvision=0.10.0, cudatoolkit=11.1

```shell
python -m pip install --upgrade pip
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Other dependencies

```shell
pip install -r requirements.txt
```
## Dataset Preparation
We use the DIV2K training split for model training, and validate on DIV2K validation split and four widely-used benchmarks: Set5, Set14, BSDS100, and Urban100.

Please organize the datasets and the code in a folder stucture as:
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

## Testing 
The pretrained models is available in `./experiments/pretrained_models` and the config files is available in `./codes/options` for quickly reproducing the results reported in the paper.

T-IRN:

For scale x2, change directory to `.code/`, run
```shell
python test.py -opt options/test/TIRN_2.yml 
```
For scale x4 , change directory to `.code/`, run
```shell
python test.py -opt options/test/TIRN_4.yml
```

T-SAIN:

For scale x2 with JPEG compression QF=90, change directory to `.code/`, run
```shell
python test.py -opt options/test/TSAIN_2.yml -format JPEG -qf 90
```
For scale x4 with JPEG compression QF=90, change directory to `.code/`, run
```shell
python test.py -opt options/test/TSAIN_4.yml -format JPEG -qf 90
```

## Training
 The training configs are included in  `./codes/options/train`. 
 
 T-IRN:
 
For scale x2, change directory to `.code/`, run
```shell
python train.py -opt options/train/TIRN_2.yml 
```
For scale x4 , change directory to `.code/`, run
```shell
python train.py -opt options/train/TIRN_4.yml
```

T-SAIN:

For scale x2 with JPEG compression QF=90, change directory to `.code/`, run
```shell
python train.py -opt options/train/TSAIN_2.yml 
```
For scale x4 with JPEG compression QF=90, change directory to `.code/`, run
```shell
python train.py -opt options/train/TSAIN_4.yml 
```
