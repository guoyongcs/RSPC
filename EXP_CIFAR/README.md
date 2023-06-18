# Improving Robustness of Vision Transformers by Reducing Sensitivity to Patch Corruptions
[Yong Guo](http://www.guoyongcs.com/), [David Stutz](https://davidstutz.de/), and [Bernt Schiele](https://scholar.google.com/citations?user=z76PBfYAAAAJ&hl=en). CVPR 2023.

### [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Improving_Robustness_of_Vision_Transformers_by_Reducing_Sensitivity_To_Patch_CVPR_2023_paper.pdf) | [Slides](https://www.guoyongcs.com/RSPC-Materials/RSPC.pdf) | [Poster](https://www.guoyongcs.com/RSPC-Materials/RSPC_Poster.pdf)



# Dependencies
Our code is built based on pytorch and timm library. Please check the detailed dependencies in [requirements.txt](https://github.com/guoyongcs/RSPC/requirements.txt).

# Dataset preparation

- Training: Please download the clean [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). We follow [DeepAugment](https://github.com/hendrycks/imagenet-r/tree/master/DeepAugment) to produce the augmented examples using the scripts [CAE_distort_imagenet.py](CAE_distort_imagenet.py) and [EDSR_distort_imagenet.py](EDSR_distort_imagenet.py). For convenience, we also provide a [link](xxx) to download the whole augmented dataset.


- Evaluation: We evaluate the model robustness on [CIFAR-10-C](https://zenodo.org/record/2535967) and [CIFAR-100-C](https://zenodo.org/record/3555552).


## Results and Pre-trained Models

### Pre-trained models on CIFAR-10 and CIFAR-100

- Pre-trained models on CIFAR-10 and CIFAR-10-C

|       Model       | CIFAR-10  | CIFAR-10-C | #Params |                                             Download                                             |
|:-----------------:|:---------:|:----------:|:-------:|:------------------------------------------------------------------------------------------------:|
|    RSPC-RVT-S     |   97.73   |   94.14    |  23.0M  | [model](https://github.com/guoyongcs/RSPC/releases/download/v1.1/rspc_rvt_small_cifar10.pth.tar) |
| RSPC-FAN-S-Hybrid | **98.06** | **94.59**  |  25.7M  | [model](https://github.com/guoyongcs/RSPC/releases/download/v1.1/rspc_fan_small_cifar10.pth.tar) |

- Pre-trained models on CIFAR-100 and CIFAR-100-C

|       Model       | CIFAR-100 | CIFAR-100-C | #Params |                                              Download                                              |
|:-----------------:|:---------:|:-----------:|:-------:|:--------------------------------------------------------------------------------------------------:|
|    RSPC-RVT-S     |   84.81   |    74.94    |    23.0M    | [model](https://github.com/guoyongcs/RSPC/releases/download/v1.1/rspc_rvt_small_cifar100.pth.tar)  |
| RSPC-FAN-S-Hybrid | **85.30** |  **75.72**  |  25.7M  | [model](https://github.com/guoyongcs/RSPC/releases/download/v1.1/rspc_fan_small_cifar100.pth.tar)  |


## Evaluation 

Evaluate RSPC-RVT-S on CIFAR-10 and CIFAR-10-C:
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model rvt_small_plus \
    --data-set CIFAR10 --data-path ../data --output_dir ../experiments/test_rspc_rvt_cifar10 \
    --cifarc_base_path ../data/CIFAR-10-C --pretrain_path ../pretrained/rspc_rvt_small_cifar10.pth.tar
```

Please see the scripts of evaluating more models in [test_pretrained.sh](test_pretrained.sh).

## Training
Train RSPC-RVT-S on CIFAR-10:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12321 main.py \
    --model rvt_small_plus --data-set CIFAR10 --data-path ../data \
    --output_dir ../experiments/exp_rspc_rvt_cifar10 \
    --deepaugment --deepaugment_base_path ../data/CIFAR-10-DeepAugment
```
Please see the scripts of training more models in [train_script.sh](train_script.sh).




## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{guo2023improving,
  title={Improving robustness of vision transformers by reducing sensitivity to patch corruptions},
  author={Guo, Yong and Stutz, David and Schiele, Bernt},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4108--4118},
  year={2023}
}
```


