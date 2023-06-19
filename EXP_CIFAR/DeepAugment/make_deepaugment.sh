# generate DeepAugment for CIFAR-10
python CAE_distort_cifar10.py --total-workers 1 --worker-number 0
python EDSR_distort_cifar10.py --total-workers 1 --worker-number 0

# generate DeepAugment for CIFAR-100
python CAE_distort_cifar100.py --total-workers 1 --worker-number 0
python EDSR_distort_cifar100.py --total-workers 1 --worker-number 0
