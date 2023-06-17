# Improving Robustness of Vision Transformers by Reducing Sensitivity to Patch Corruptions
[Yong Guo](http://www.guoyongcs.com/), [David Stutz](https://davidstutz.de/), and [Bernt Schiele](https://scholar.google.com/citations?user=z76PBfYAAAAJ&hl=en). CVPR 2023.

### [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Improving_Robustness_of_Vision_Transformers_by_Reducing_Sensitivity_To_Patch_CVPR_2023_paper.pdf) | [Slides](https://www.guoyongcs.com/RSPC-Materials/RSPC.pdf) | [Poster](https://www.guoyongcs.com/RSPC-Materials/RSPC_Poster.pdf)


<p align="center">
<img src="demo/Teaser.png" width=60% height=60% 
class="center">
</p>

This repository contains the official Pytorch implementation and the pretrained models of [Reducing Sensitivity to Patch Corruptions](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Improving_Robustness_of_Vision_Transformers_by_Reducing_Sensitivity_To_Patch_CVPR_2023_paper.pdf) (**RSPC**).


## Catalog
- [x] Pre-trained Models on CIFAR
- [x] Pre-trained Models on ImageNet
- [x] Evaluation and Training Code




# Dependencies
Our code is built based on timm library, which can be installed via:
pip3 install timm==0.7.0.dev0
pip3 install torchvision==0.9.1.

Please check the other dependencies in [requirements.txt](requirements.txt).

# Dataset preparation
Download [ImageNet](http://image-net.org/) clean dataset and [ImageNet-C](https://zenodo.org/record/2235448) dataset and structure the datasets as follows:

```
/PATH/TO/IMAGENET-C/
  clean/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
  corruption1/
    severity1/
      class1/
        img3.jpeg
      class2/
        img4.jpeg
    severity2/
      class1/
        img3.jpeg
      class2/
        img4.jpeg
```

