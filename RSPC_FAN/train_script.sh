# train RSPC-FAN-T on ImageNet (4 nodes and each with 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py /PATH/TO/IMAGENET --model fan_tiny_8_p4_hybrid -b 128 --sched cosine --epochs 300 --opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5 --model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.3 --lr 20e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 --drop-path .1 --img-size 224 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --output ../experiments/exp_rspc_fan_tiny_imagenet --amp --model-ema


# train RSPC-FAN-S on ImageNet (8 nodes and each with 8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py /PATH/TO/IMAGENET --model fan_small_12_p4_hybrid -b 64 --sched cosine --epochs 300 --opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 5 --model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.3 --lr 40e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 --drop-path .25 --img-size 224 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --output ../experiments/exp_rspc_fan_small_imagenet --amp --model-ema

# train RSPC-FAN-B on ImageNet (8 nodes and each with 8 GPUs)
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py /PATH/TO/IMAGENET --model fan_base_16_p4_hybrid -b 64 --sched cosine --epochs 300 --opt adamw -j 16 --warmup-lr 1e-6 --warmup-epochs 10 --model-ema-decay 0.99992 --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.3 --lr 40e-4 --min-lr 1e-6 --weight-decay .05 --drop 0.0 --drop-path .35 --img-size 224 --mixup 0.8 --cutmix 1.0 --smoothing 0.1 --output ../experiments/exp_rspc_fan_base_imagenet --amp --model-ema