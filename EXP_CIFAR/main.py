# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torchvision
import json
import shutil

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model, load_checkpoint, resume_checkpoint
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEmaV2

from datasets import build_dataset, build_transform
from engine import train_one_epoch, train_one_epoch_rspc, evaluate, eval_inc, eval_cifarc
from losses import DistillationLoss
from samplers import RASampler
import robust_models
import fan_models
import utils
import logging
import glob
import sys
import torch.distributed as dist


def get_args_parser():
    parser = argparse.ArgumentParser('RSPC training and evaluation script for RVT', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='rvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # RVT params
    parser.add_argument('--use_patch_aug', action='store_false')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='load pretrained model')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # eval parameters
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--inc_path', default=None, type=str, help='imagenet-c')
    parser.add_argument('--ina_path', default=None, type=str, help='imagenet-a')
    parser.add_argument('--inr_path', default=None, type=str, help='imagenet-r')
    parser.add_argument('--insk_path', default=None, type=str, help='imagenet-sketch')
    parser.add_argument('--cifarc_base_path', default=None, type=str, help='cifarc_base_path')
    parser.add_argument('--fgsm_test', action='store_true', default=False, help='test on FGSM attacker')
    parser.add_argument('--pgd_test', action='store_true', default=False, help='test on PGD attacker')

    parser.add_argument('--dist-eval', action='store_false', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # DeepAugment
    parser.add_argument('--deepaugment', action='store_true', default=False, help='deepaugment')
    parser.add_argument('--deepaugment_base_path', type=str, default=None, help='deepaugment_base_path')
    parser.add_argument('--remove_deepaug', type=int, default=1000, help='remove_deepaug')

    # FT params
    parser.add_argument('--pretrain_path', type=str, default=None, help='pretrain_path')

    # afat parameters
    parser.add_argument('--occlusion_ratio', default=0.1, type=float, help="occlusion ratio")
    parser.add_argument('--afa', action='store_true', help='adversarial feature alignment')
    parser.add_argument('--no_rspc', action='store_false', dest='use_rspc', help='without RSPC')
    parser.add_argument('--extra_weight', type=float, default=0.005, help='lambda: the importance of feature alignment loss')
    parser.add_argument('--eval_model_ema', action='store_true', help='eval_model_ema')

    return parser

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl'
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('| distributed init {}(rank {})'.format(
                args.world_size, args.rank), flush=True)
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)
    else:
        print('Not using distributed mode')
        args.distributed = False

    # create logger
    log_format = '%(asctime)s %(message)s'
    dist_rank = dist.get_rank() if utils.is_dist_avail_and_initialized() else 0
    logger = logging.getLogger('EWS')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # create console handlers for master process
    if dist_rank == 0:
        print(f'setting up console logger {dist_rank}')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=log_format, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    print(f'setting up file logger {dist_rank}')
    file_handler = logging.FileHandler(os.path.join(args.output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    logger.info(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)

    occlusion_model = None
    occlusion_model_optimizer = None
    if args.use_rspc:
        occlusion_model = robust_models.PatchOcclusion(args.occlusion_ratio, 3, 16, 196)
        occlusion_model.to(device)
        occlusion_model_optimizer = torch.optim.SGD(occlusion_model.parameters(), momentum=0.9, nesterov=True, lr=1e-4, weight_decay=0.05)

    loss_scaler = NativeScaler()

    lr_scheduler, args.epochs = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        logger.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)

    checkpoint_path = os.path.join(args.output_dir, 'last.pth.tar')
    if os.path.exists(checkpoint_path):
        args.resume = checkpoint_path
    else:
        if args.pretrain_path is not None:
            logger.info(f'Load pretrain_path from {args.pretrain_path}')
            load_checkpoint(model, args.pretrain_path, use_ema=args.eval_model_ema)
            load_checkpoint(model_ema.module, args.pretrain_path, use_ema=args.eval_model_ema)
            if occlusion_model is not None:
                try:
                    _ = load_checkpoint(occlusion_model, os.path.join(os.path.dirname(args.pretrain_path), 'occlusion', 'last.pth.tar'))
                except:
                    pass

    saver = None
    occlusion_saver = None
    if dist_rank == 0:
        saver = utils.MyCheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=args.output_dir, recovery_dir=args.output_dir, decreasing=False, max_history=10)
        if occlusion_model is not None:
            save_path_occlusion = os.path.join(args.output_dir, 'occlusion')
            occlusion_saver = utils.MyCheckpointSaver(
                model=occlusion_model, optimizer=occlusion_model_optimizer, args=args, checkpoint_dir=save_path_occlusion, recovery_dir=save_path_occlusion, decreasing=False, max_history=10)
        if args.resume:
            try:
                saver.load_checkpoint_files()
            except:
                print('skip loading checkpoint_files')
            if occlusion_saver is not None:
                occlusion_saver.load_checkpoint_files()

    if args.resume:
        args.start_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            log_info=args.local_rank == 0)
        load_checkpoint(model_ema.module, args.resume, use_ema=True)

        if occlusion_model is not None:
            _ = resume_checkpoint(
                occlusion_model, os.path.join(args.output_dir, 'occlusion', 'last.pth.tar'),
                optimizer=occlusion_model_optimizer,
                log_info=args.local_rank == 0)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        if occlusion_model is not None:
            occlusion_model = torch.nn.parallel.DistributedDataParallel(occlusion_model, device_ids=[args.local_rank])

    if args.eval:
        test_stats = evaluate(logger, data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if args.cifarc_base_path:
            eval_cifarc(logger, model, torch.device('cuda'), args)
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.use_rspc:
            train_stats = train_one_epoch_rspc(logger, args, model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn, set_training_mode=args.finetune == '', occlusion_model=occlusion_model, occlusion_model_optimizer=occlusion_model_optimizer)
        else:
            train_stats = train_one_epoch(
                logger, args, model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
            )

        lr_scheduler.step(epoch)

        test_stats = evaluate(logger, data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if model_ema is not None and not args.model_ema_force_cpu:
            try:
                ema_eval_metrics = evaluate(logger, data_loader_val, model_ema.module, device)
                test_stats['ema_acc'] = ema_eval_metrics['acc1']
            except:
                pass

        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = max(test_stats['acc1'], test_stats['ema_acc']) if 'ema_acc' in test_stats.keys() else test_stats['acc1']
            max_accuracy, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
            if test_stats['ema_acc']>test_stats['acc1']:
                args.eval_model_ema = True

            saver.save_checkpoint_files()
            if occlusion_saver is not None:
                _, _ = occlusion_saver.save_checkpoint(epoch, metric=save_metric)
                occlusion_saver.save_checkpoint_files()
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RSPC training and evaluation script for RVT', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(os.path.join(args.output_dir, 'occlusion')).mkdir(parents=True, exist_ok=True)
    main(args)
