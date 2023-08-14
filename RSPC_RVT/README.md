# Training and Evaluation of RSPC-RVT on ImageNet
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



## Results and Pre-trained Models of RSPC-RVT


|       Model       | IN-1K $\uparrow$ | IN-C $\downarrow$ | IN-A $\uparrow$ | IN-P $\downarrow$ | #Params |                                                                                Download                                                                                |
|:-----------------:|:----------------:|:-----------------:|:---------------:|:-----------------:|:-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    RSPC-RVT-Ti    |       79.5       |       55.7        |      16.5       |       38.0        |    10.9M    |                                        [model](https://github.com/guoyongcs/RSPC/releases/download/v1.0/rspc_rvt_tiny.pth.tar)                                         |
|    RSPC-RVT-S     |       82.2       |       48.4        |      27.9       |       34.3        |  23.3M  |                                        [model](https://github.com/guoyongcs/RSPC/releases/download/v1.0/rspc_rvt_small.pth.tar)                                        |
|    RSPC-RVT-B     |     **82.8**     |     **45.7**      |    **32.1**     |     **31.0**      |  91.8M  |                                        [model](https://github.com/guoyongcs/RSPC/releases/download/v1.0/rspc_rvt_base.pth.tar)                                         |




## Evaluation

Evaluate RSPC-RVT-Ti on ImageNet (and optionally on ImageNet-C):
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model rvt_tiny_plus \
    --data-set IMNET --data-path /PATH/TO/IMAGENET --output_dir ../experiments/test_rspc_rvt_tiny_imagenet \
    --pretrain_path ../pretrained/rspc_rvt_tiny.pth.tar --inc_path /PATH/TO/IMAGENET-C
```
Please see the scripts of evaluating more models in [test_pretrained.sh](test_pretrained.sh).


## Training 
Train RSPC-RVT-Ti on ImageNet (using 4 nodes and each with 4 GPUs)
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus_afat \
    --data-path /PATH/TO/IMAGENET --output_dir ../experiments/exp_rspc_rvt_tiny_imagenet \
    --batch-size 128 --dist-eval --use_patch_aug
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


