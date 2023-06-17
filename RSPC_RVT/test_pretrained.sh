# test RSPC-RVT-T on ImageNet
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model rvt_tiny_plus --data-set IMNET --data-path /PATH/TO/IMAGENET --output_dir ../experiments/test_rspc_rvt_tiny_imagenet --pretrain_path ../pretrained/rspc_rvt_tiny.pth.tar --inc_path /PATH/TO/IMAGENET-C

# test RSPC-RVT-S on ImageNet
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model rvt_small_plus --data-set IMNET --data-path /PATH/TO/IMAGENET --output_dir ../experiments/test_rspc_rvt_small_imagenet --pretrain_path ../pretrained/rspc_rvt_small.pth.tar --inc_path /PATH/TO/IMAGENET-C

# test RSPC-RVT-B on ImageNet
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model rvt_base_plus --data-set IMNET --data-path /PATH/TO/IMAGENET --output_dir ../experiments/test_rspc_rvt_base_imagenet --pretrain_path ../pretrained/rspc_rvt_base.pth.tar --inc_path /PATH/TO/IMAGENET-C

