# VoxHRNet

This is the official implementation of the following paper:

> [**Whole Brain Segmentation with Full Volume Neural Network**](https://doi.org/10.1016/j.compmedimag.2021.101991)
> 
> Yeshu Li, Jonathan Cui, Yilun Sheng, Xiao Liang, Jingdong Wang, Eric I-Chao Chang, Yan Xu
> 
> *[Computerized Medical Imaging and Graphics](https://doi.org/10.1016/j.compmedimag.2021.101991)*
> 
> [[arXiv]](https://arxiv.org/abs/2110.15601)

## Network

![architecture](https://ars.els-cdn.com/content/image/1-s2.0-S0895611121001403-gr2_lrg.jpg)

## Installation

The following environments/libraries are required:
- Python 3
- yacs
- SimpleITK
- apex
- pytorch
- nibabel
- numpy
- scikit-image
- scipy

## Quick Start

### Data Preparation

Download the [LPBA40](https://resource.loni.usc.edu/resources/atlases-downloads/) and [Hammers n30r95](https://brain-development.org/brain-atlases/adult-brain-atlases/) datasets.

After renaming, your directory tree should look like:
```bash
$ROOT
├── data
│   └── LPBA40_N4_RN
│       ├── aseg_TEST001.nii.gz
│       ├── ...
│       ├── aseg_TEST010.nii.gz
│       ├── aseg_TRAIN001.nii.gz
│       ├── ...
│       ├── aseg_TRAIN027.nii.gz
│       ├── aseg_VALIDATE001.nii.gz
│       ├── ...
│       ├── aseg_VALIDATE003.nii.gz
│       ├── orig_TEST001.nii.gz
│       ├── ...
│       ├── orig_TEST010.nii.gz
│       ├── orig_TRAIN001.nii.gz
│       ├── ...
│       ├── orig_TRAIN027.nii.gz
│       ├── orig_VALIDATE001.nii.gz
│       ├── ...
│       └── orig_VALIDATE003.nii.gz
└── VoxHRNet
    ├── voxhrnet.py
    ├── ...
    └── train.py
```

Create a [YACS](https://github.com/rbgirshick/yacs) configuration file and make changes for specific training/test settings accordingly. We use `config_lpba.yaml` as an example as follows.

### Train

Run

```shell
python3 train.py --cfg config_lpba.yaml
```

### Test

Run

```shell
python3 test.py --cfg config_lpba.yaml
```

## Pretrained Models

For the [LPBA40](https://resource.loni.usc.edu/resources/atlases-downloads/) dataset, we number the subjects from 1-40 alphabetically and split them into 4 folds sequentially. The k-th fold is selected as the test set in the k-th split.

For the [Hammers n30r95](https://brain-development.org/brain-atlases/adult-brain-atlases/) dataset, the first 20 subjects and last 10 subjects are chosen as the training and test set respectively.

Their pretrained models can be found in the [release](https://github.com/microsoft/VoxHRNet/releases) page of this repository.

## Citation

Please cite our work if you find it useful in your research:

```
@article{LI2021101991,
title = {Whole brain segmentation with full volume neural network},
journal = {Computerized Medical Imaging and Graphics},
volume = {93},
pages = {101991},
year = {2021},
issn = {0895-6111},
doi = {https://doi.org/10.1016/j.compmedimag.2021.101991},
url = {https://www.sciencedirect.com/science/article/pii/S0895611121001403},
author = {Yeshu Li and Jonathan Cui and Yilun Sheng and Xiao Liang and Jingdong Wang and Eric I.-Chao Chang and Yan Xu},
keywords = {Brain, Segmentation, Neural networks, Deep learning},
abstract = {Whole brain segmentation is an important neuroimaging task that segments the whole brain volume into anatomically labeled regions-of-interest. Convolutional neural networks have demonstrated good performance in this task. Existing solutions, usually segment the brain image by classifying the voxels, or labeling the slices or the sub-volumes separately. Their representation learning is based on parts of the whole volume whereas their labeling result is produced by aggregation of partial segmentation. Learning and inference with incomplete information could lead to sub-optimal final segmentation result. To address these issues, we propose to adopt a full volume framework, which feeds the full volume brain image into the segmentation network and directly outputs the segmentation result for the whole brain volume. The framework makes use of complete information in each volume and can be implemented easily. An effective instance in this framework is given subsequently. We adopt the 3D high-resolution network (HRNet) for learning spatially fine-grained representations and the mixed precision training scheme for memory-efficient training. Extensive experiment results on a publicly available 3D MRI brain dataset show that our proposed model advances the state-of-the-art methods in terms of segmentation performance.}
}
```

## Acknowledgement

A large part of the code is borrowed from [HRNet](https://github.com/HRNet).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
