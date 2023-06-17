# train RSPC-RVT-S on CIFAR10
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12321 main.py --model rvt_small_plus --data-set CIFAR10 --data-path ../data --output_dir ../experiments/exp_rspc_rvt_cifar10 --deepaugment --deepaugment_base_path ../data/CIFAR-10-DeepAugment

# train RSPC-RVT-S on CIFAR100
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12322 main.py --model rvt_small_plus --data-set CIFAR100 --data-path ../data --output_dir ../experiments/exp_rspc_rvt_cifar100 --deepaugment --deepaugment_base_path ../data/CIFAR-100-DeepAugment

# train RSPC-FAN-S on CIFAR10
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12323 main.py --model fan_small_12_p4_hybrid --data-set CIFAR10 --data-path ../data --output_dir ../experiments/exp_rspc_fan_cifar10 --deepaugment --deepaugment_base_path ../data/CIFAR-100-DeepAugment

# train RSPC-FAN-S on CIFAR100
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12324 main.py --model fan_small_12_p4_hybrid --data-set CIFAR100 --data-path ../data --output_dir ../experiments/exp_rspc_fan_cifar100 --deepaugment --deepaugment_base_path ../data/CIFAR-100-DeepAugment

python -m torch.distributed.launch --nproc_per_node=2 --master_port 12321 main.py --model rvt_small_plus --data-set CIFAR10 --data-path ../data --output_dir ../experiments/exp_rspc_rvt_cifar10 --deepaugment --deepaugment_base_path ../data/CIFAR-10-DeepAugment
