"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import kornia as K

from losses import DistillationLoss
import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision
from datasets import build_dataset


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def patch_level_aug(input1, patch_transform, upper_limit, lower_limit):
    bs, channle_size, H, W = input1.shape
    patches = input1.unfold(2, 16, 16).unfold(3, 16, 16).permute(0,2,3,1,4,5).contiguous().reshape(-1, channle_size,16,16)
    patches = patch_transform(patches)
 
    patches = patches.reshape(bs, -1, channle_size,16,16).permute(0,2,3,4,1).contiguous().reshape(bs, channle_size*16*16, -1)
    output_images = F.fold(patches, (H,W), 16, stride=16)
    output_images = clamp(output_images, lower_limit, upper_limit)
    return output_images


def train_one_epoch(logger, args, model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
    mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, logger, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16,16), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
                )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        use_autocast = False if 'rvt_base' in args.model else True
        with torch.cuda.amp.autocast(enabled=use_autocast):
            if args.use_patch_aug:
                outputs2 = model(aug_samples)
                loss = criterion(aug_samples, outputs2, targets)
                loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
            else:
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, disable autocast".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_rspc(logger, args, model: torch.nn.Module, criterion: DistillationLoss, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, set_training_mode=True, occlusion_model=None, occlusion_model_optimizer=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
    mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, logger, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(16,16), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
            )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        use_autocast = False if 'rvt_base' in args.model else True
        with torch.cuda.amp.autocast(enabled=use_autocast):
            if args.use_patch_aug:
                # generate occluded samples for patchaug
                noise_mask = occlusion_model(aug_samples)
                noise = torch.rand(aug_samples.shape, dtype=aug_samples.dtype, device=aug_samples.device)
                noise_aug_sample = aug_samples * noise_mask + noise * (1-noise_mask)
                noise_aug_sample = clamp(noise_aug_sample, lower_limit, upper_limit)
                mask_outputs2, outputs2 = model(noise_aug_sample, aug_samples)
                loss = criterion(samples, outputs2, targets)
                if args.extra_weight > 0:
                    extra_loss = utils.FeatureAlignmentLoss(model)
                    loss += args.extra_weight * extra_loss
                loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)
                # generate occluded samples
                noise_mask = occlusion_model(samples)
                noise = torch.rand(samples.shape, dtype=samples.dtype, device=samples.device)
                noise_sample = samples * noise_mask + noise * (1-noise_mask)
                noise_sample = clamp(noise_sample, lower_limit, upper_limit)
                mask_outputs, outputs = model(noise_sample, samples)
                loss = criterion(samples, outputs, targets)
                if args.extra_weight > 0:
                    extra_loss = utils.FeatureAlignmentLoss(model)
                    loss += args.extra_weight * extra_loss
            else:
                # generate occluded samples
                noise_mask = occlusion_model(samples)
                noise = torch.rand(samples.shape, dtype=samples.dtype, device=samples.device)
                noise_sample = samples * noise_mask + noise * (1-noise_mask)
                noise_sample = clamp(noise_sample, lower_limit, upper_limit)
                mask_outputs, outputs = model(noise_sample, samples)
                loss = criterion(samples, outputs, targets)
                if args.extra_weight > 0:
                    extra_loss = utils.FeatureAlignmentLoss(model)
                    loss += args.extra_weight * extra_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, disable autocast".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        occlusion_model_optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
        # update the occlusion model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            torch.nn.utils.clip_grad_norm_(occlusion_model.module.parameters(), 5)
        else:
            torch.nn.utils.clip_grad_norm_(occlusion_model.parameters(), 5)
        utils.reverse_grad(occlusion_model_optimizer)
        occlusion_model_optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(logger, data_loader, model, device, mask=None, indices_in_1k=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            if indices_in_1k is not None:
                output = model(images)[:,indices_in_1k]
            else:
                output = model(images)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_cifarc(logger, model, device, args=None):
    """Evaluate network on given corrupted dataset."""
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]

    # switch to evaluation mode
    model.eval()

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    test_data, _ = build_dataset(False, args)
    base_path = args.cifarc_base_path

    corruption_accs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(os.path.join(base_path, corruption + '.npy'))
        test_data.targets = torch.LongTensor(np.load(os.path.join(base_path, 'labels.npy')))

        sampler_val = torch.utils.data.DistributedSampler(
            test_data, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        test_loader_val = torch.utils.data.DataLoader(
            test_data, sampler=sampler_val,
            batch_size=int(args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        test_stats = evaluate(logger, test_loader_val, model, device)
        corruption_accs.append(test_stats['acc1'])
    logger.info(f'mean corruption accuracy: {np.mean(corruption_accs):.2f}')

    return np.mean(corruption_accs)

