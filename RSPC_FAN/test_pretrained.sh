# test RSPC-FAN-T on ImageNet
CUDA_VISIBLE_DEVICES=0 python main.py /PATH/TO/IMAGENET --model fan_tiny_8_p4_hybrid --output ../experiments/test_rspc_fan_tiny_imagenet --amp --eval-first --pretrain_path ../pretrained/rspc_fan_tiny.pth.tar --inc_path /PATH/TO/IMAGENET-C

# test RSPC-FAN-S on ImageNet
CUDA_VISIBLE_DEVICES=0 python main.py /BS/database11/ILSVRC2012 --model fan_small_12_p4_hybrid --output ../experiments/test_rspc_fan_small_imagenet --amp --eval-first --pretrain_path ../pretrained/rspc_fan_small.pth.tar --inc_path /PATH/TO/IMAGENET-C

# test RSPC-FAN-B on ImageNet
CUDA_VISIBLE_DEVICES=0 python main.py /PATH/TO/IMAGENET --model fan_base_16_p4_hybrid --output ../experiments/test_rspc_fan_base_imagenet --amp --eval-first --pretrain_path ../pretrained/rspc_fan_base.pth.tar --inc_path /PATH/TO/IMAGENET-C



