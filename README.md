# Referring segmentation framework


## Summary
| Code     |Backbone  | Paper |  Journal |  Year | 
| ----------- | ----------- | ----------- |----------- |----------- |
| [LAVT](https://github.com/yz93/LAVT-RIS)     | Swin+Bert | [LAVT: Language-Aware Vision Transformer for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_LAVT_Language-Aware_Vision_Transformer_for_Referring_Image_Segmentation_CVPR_2022_paper.html)       | CVPR       | 2022       | 
| [RRSIS](https://gitlab.lrz.de/ai4eo/reasoning/rrsis)     | Swin+Bert | [RRSIS: Referring Remote Sensing  Image Segmentation](https://ieeexplore.ieee.org/abstract/document/10458079/)       | TGRS       | 2024       | 
| [RMSIN](https://github.com/Lsan2401/RMSIN)     | Swin+Bert | [Rotated Multi-Scale Interaction Network for Referring Remote Sensing Image Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Rotated_Multi-Scale_Interaction_Network_for_Referring_Remote_Sensing_Image_Segmentation_CVPR_2024_paper.html)       | CVPR       | 2024       | 

This repository provides the a collection of implementations for refering segmentation. I hopt it could be a good start for this task. 

The original rely on mmcv and mmsegmentation. I found they are unnecessary, so I remove all of them and keep this version simple. I hope this version of the code will work in the latest python and pytorch environment.

I ran the code to make sure the model achieve the little difference results comparing to original resutls.


## Files
Code in this repository is written using [PyTorch](https://pytorch.org/) and is organized in the following way (assuming the working directory is the root directory of this repository):
* `./lib` contains files implementing the main network.
* Inside `./lib`, `_utils.py` defines the highest-level model, which incorporates the backbone network
defined in `backbone.py` and the simple mask decoder defined in `mask_predictor.py`.
`segmentation.py` provides the model interface and initialization functions.
* `./bert` contains files migrated from [Hugging Face Transformers v3.0.2](https://huggingface.co/transformers/v3.0.2/quicktour.html),
which implement the BERT language model.
We used Transformers v3.0.2 during development but it had a bug that would appear when using `DistributedDataParallel`.
Therefore we maintain a copy of the relevant source files in this repository.
This way, the bug is fixed and code in this repository is self-contained.
* `./train.py` is invoked to train the model.
* `./test.py` is invoked to run inference on the evaluation subsets after training.
* `./refer` contains data pre-processing code and is also where data should be placed, including the images and all annotations.
It is cloned from [refer](https://github.com/lichengunc/refer). 
* `./data/dataset_refer_bert.py` is where the dataset class is defined.
* `./utils.py` defines functions that track training statistics and setup
* `./loss.py` defines loss functions

functions for `DistributedDataParallel`.


## Setting Up


Weights are needed for training to initialize the model.

1. Download pre-trained classification weights of the [ Swin Transformer](https://github.com/microsoft/Swin-Transformer), i.e. [swin_base_patch4_window12_384_22k](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)
and put the `pth` file in `./swin`.
2. Download [bert weights](https://huggingface.co/google-bert/bert-base-uncased/tree/main),**config.json, pytorch_model.bin, tokenizer_config.json, vocab.txt** 
and put these files in `./bert-base-uncased`.

Path are needed for datasets load and weights save. In the args.py file, revise the following path:
1. ck_bert: bert weights
2. pretrained_swin_weights: swin transformer weights
3. refer_data_root: dataset path
4. output-dir: output weights path
5. resume: resume weights path


## Datasets
- RRSIS-D: It can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Xqi3Am2Vgm4a5tHqiV9tfaqKNovcuK3A?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1yZatV2w_bSXIP9QBv2lCrA?pwd=sjoe) (access code: sjoe).

### Usage
1. Download our dataset.
2. Copy all the downloaded files to `./refer/data/`. The dataset folder should be like this:
```
$DATA_PATH
├── rrsisd
│   ├── refs(unc).p
│   ├── instances.json
└── images
    └── rrsisd
        ├── JPEGImages
        ├── ann_split

```
   
## Training
We use DistributedDataParallel from PyTorch for training. To run on 4 GPUs (with IDs 0, 1, 2, and 3) on a single node:
```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 train.py --dataset rrsisd --model_id RMSIN --epochs 40 --img_size 480 

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 train.py
```

## Testing
```shell
python test.py --swin_type base --dataset rrsisd --resume ./your_checkpoints_path --split val --workers 4 --window12 --img_size 480
```

## Acknowledgements
The code mainly is built on [LAVT](https://github.com/yz93/LAVT-RIS) and [RMSIN](https://github.com/Lsan2401/RMSIN). We'd like to thank the authors for open sourcing their project.
