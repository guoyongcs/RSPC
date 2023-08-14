# Training and Evaluation of RSPC-FAN on ImageNet
[Improving Robustness of Vision Transformers by Reducing Sensitivity to Patch Corruptions](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Improving_Robustness_of_Vision_Transformers_by_Reducing_Sensitivity_To_Patch_CVPR_2023_paper.pdf), \
[Yong Guo](http://www.guoyongcs.com/), [David Stutz](https://davidstutz.de/), and [Bernt Schiele](https://scholar.google.com/citations?user=z76PBfYAAAAJ&hl=en). CVPR 2023.



# Dependencies
Our code is built based on pytorch and timm library. Please check the detailed dependencies in [requirements.txt](https://github.com/guoyongcs/RSPC/blob/main/requirements.txt).

# Dataset Preparation

Please download the clean [ImageNet](http://image-net.org/) dataset and [ImageNet-C](https://zenodo.org/record/2235448) dataset and structure the datasets as follows:

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

We also use other robustness benchmarks for evaluation, including [ImageNet-A](https://github.com/hendrycks/natural-adv-examples) and [ImageNet-P](https://zenodo.org/record/3565846).



## Results and Pre-trained Models of RSPC-FAN

|       Model       | IN-1K $\uparrow$ | IN-C $\downarrow$ | IN-A $\uparrow$ | IN-P $\downarrow$ | #Params |                                         Download                                         |
|:-----------------:|:----------------:|:-----------------:|:---------------:|:-----------------:|:-------:|:----------------------------------------------------------------------------------------:|
| RSPC-FAN-T-Hybrid |       80.3       |       57.2        |      23.6       |       37.3        |    7.5M    | [model](https://github.com/guoyongcs/RSPC/releases/download/v1.0/rspc_fan_tiny.pth.tar)  |
| RSPC-FAN-S-Hybrid |       83.6       |       47.5        |      36.8       |       33.5        |  25.7M  | [model](https://github.com/guoyongcs/RSPC/releases/download/v1.0/rspc_fan_small.pth.tar) |
|  RSPC-FAN-B-ViT   |     **84.2**     |     **44.5**      |    **41.1**     |     **30.0**      |  50.5M  | [model](https://github.com/guoyongcs/RSPC/releases/download/v1.0/rspc_fan_base.pth.tar)  |


## Evaluation
- Evaluate RSPC-FAN-T-Hybrid on ImageNet (and optionally on ImageNet-C):
```
CUDA_VISIBLE_DEVICES=0 python main.py /PATH/TO/IMAGENET --model fan_tiny_8_p4_hybrid \
    --output ../experiments/test_rspc_fan_tiny_imagenet --amp --eval-first \
    --pretrain_path ../pretrained/rspc_fan_tiny.pth.tar --inc_path /PATH/TO/IMAGENET-C
```
Please see the scripts of evaluating more models in [test_pretrained.sh](test_pretrained.sh).

- Evaluate on other robustness benchmarks: Please use [validate_ood.py](validate_ood.py) and refer to [FAN](https://github.com/NVlabs/FAN).

## Training 
Train RSPC-FAN-T on ImageNet (using 4 nodes and each with 4 GPUs)
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py /PATH/TO/IMAGENET \
    --model fan_tiny_8_p4_hybrid -b 128 --sched cosine --epochs 300 --opt adamw -j 16 \
    --warmup-lr 1e-6 --warmup-epochs 5 --model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 \
    --remode pixel --reprob 0.3 --lr 20e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 --drop-path .1 \
    --img-size 224 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --output ../experiments/exp_rspc_fan_tiny_imagenet \
    --amp --model-ema
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


