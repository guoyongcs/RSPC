# test RSPC-RVT-S on CIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model rvt_small_plus --data-set CIFAR10 --data-path ../data --output_dir ../experiments/test_rspc_rvt_cifar10 --cifarc_base_path ../data/CIFAR-10-C --pretrain_path ../pretrained/rspc_rvt_small_cifar10.pth.tar

# test RSPC-RVT-S on CIFAR100
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model rvt_small_plus --data-set CIFAR100 --data-path ../data --output_dir ../experiments/test_rspc_rvt_cifar100 --cifarc_base_path ../data/CIFAR-100-C --pretrain_path ../pretrained/rspc_rvt_small_cifar100.pth.tar

# test RSPC-FAN-S on CIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model fan_small_12_p4_hybrid --data-set CIFAR10 --data-path ../data --output_dir ../experiments/test_rspc_fan_cifar10 --cifarc_base_path ../data/CIFAR-10-C --pretrain_path ../pretrained/rspc_fan_small_cifar10.pth.tar

# test RSPC-FAN-S on CIFAR100
CUDA_VISIBLE_DEVICES=1 python main.py --eval --model fan_small_12_p4_hybrid --data-set CIFAR100 --data-path ../data --output_dir ../experiments/test_rspc_fan_cifar100 --cifarc_base_path ../data/CIFAR-100-C --pretrain_path ../pretrained/rspc_fan_small_cifar100.pth.tar
