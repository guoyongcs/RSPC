# train RSPC-RVT-T on ImageNet (4 nodes and each with 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_tiny_plus_afat --data-path /PATH/TO/IMAGENET --output_dir ../experiments/exp_rspc_rvt_tiny_imagenet --batch-size 128 --dist-eval --use_patch_aug


# train RSPC-FAN-S on ImageNet (4 nodes and each with 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_small_plus --data-path /PATH/TO/IMAGENET --output_dir ../experiments/exp_rspc_rvt_small_imagenet --batch-size 128 --dist-eval --use_patch_aug

# train RSPC-FAN-B on ImageNet (8 nodes and each with 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py --model rvt_base_plus_afat --data-path /PATH/TO/IMAGENET --output_dir ../experiments/exp_rspc_rvt_base_imagenet --batch-size 64 --dist-eval --use_patch_aug
