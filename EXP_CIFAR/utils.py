# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
from robust_models import Block, MaskBlock, Attention, DecompsableAttention
import torch.nn as nn
from timm.utils import *
from fan_models.fan import FANBlock


data_loaders_names = {
    'Brightness': 'brightness',
    'Contrast': 'contrast',
    'Defocus Blur': 'defocus_blur',
    'Elastic Transform': 'elastic_transform',
    'Fog': 'fog',
    'Frost': 'frost',
    'Gaussian Noise': 'gaussian_noise',
    'Glass Blur': 'glass_blur',
    'Impulse Noise': 'impulse_noise',
    'JPEG Compression': 'jpeg_compression',
    'Motion Blur': 'motion_blur',
    'Pixelate': 'pixelate',
    'Shot Noise': 'shot_noise',
    'Snow': 'snow',
    'Zoom Blur': 'zoom_blur'
}

def get_ce_alexnet():
    """Returns Corruption Error values for AlexNet"""

    ce_alexnet = dict()
    ce_alexnet['Gaussian Noise'] = 0.886428
    ce_alexnet['Shot Noise'] = 0.894468
    ce_alexnet['Impulse Noise'] = 0.922640
    ce_alexnet['Defocus Blur'] = 0.819880
    ce_alexnet['Glass Blur'] = 0.826268
    ce_alexnet['Motion Blur'] = 0.785948
    ce_alexnet['Zoom Blur'] = 0.798360
    ce_alexnet['Snow'] = 0.866816
    ce_alexnet['Frost'] = 0.826572
    ce_alexnet['Fog'] = 0.819324
    ce_alexnet['Brightness'] = 0.564592
    ce_alexnet['Contrast'] = 0.853204
    ce_alexnet['Elastic Transform'] = 0.646056
    ce_alexnet['Pixelate'] = 0.717840
    ce_alexnet['JPEG Compression'] = 0.606500

    return ce_alexnet

def get_mce_from_accuracy(accuracy, error_alexnet):
    """Computes mean Corruption Error from accuracy"""
    error = 100. - accuracy
    ce = error / (error_alexnet * 100.)

    return ce

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, logger, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    new_checkpoint = {'state_dict_ema': checkpoint}
    mem_file = io.BytesIO()
    torch.save(new_checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def FeatureAlignmentLoss(model):
    loss = 0
    counter = 1e-4
    for module in model.modules():
        if isinstance(module, (Block, MaskBlock, FANBlock)):
            if module.recon_loss is not None:
                loss += module.recon_loss
                counter += 1
    loss = loss / counter
    return loss


def reverse_grad(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                tmp_grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                p.grad = -1*tmp_grad


def compute_flops(module: nn.Module, size, skip_pattern):
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, Attention, DecompsableAttention)):
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size))
        module.train(mode=training)
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(m, nn.Linear):
            N, _ = m.output_size
            flops += m.in_features * m.out_features * N
        if isinstance(m, Attention):
            N, C = m.output_size
            flops += N * C * 3 * C
            flops += N * C * N * 2
            flops += N * C * C
        if isinstance(m, DecompsableAttention):
            N, C = m.output_size
            flops += N * C * 3 * C
            flops += N * C * 2 * 2
            flops += N * C * C
    return flops


class MyCheckpointSaver(CheckpointSaver):
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model):

        super().__init__(model,
                         optimizer,
                         args,
                         model_ema,
                         amp_scaler,
                         checkpoint_prefix,
                         recovery_prefix,
                         checkpoint_dir,
                         recovery_dir,
                         decreasing,
                         max_history,
                         unwrap_fn)

    def load_checkpoint_files(self):
        tmp_save_path = os.path.join(self.checkpoint_dir, 'checkpoint_files' + self.extension)
        file_to_load = torch.load(tmp_save_path)
        self.checkpoint_files = file_to_load['checkpoint_files']
        self.best_metric = file_to_load['best_metric']

    def save_checkpoint_files(self):
        tmp_save_path = os.path.join(self.checkpoint_dir, 'checkpoint_files' + self.extension)
        file_to_save = {'checkpoint_files': self.checkpoint_files, 'best_metric': self.best_metric}
        torch.save(file_to_save, tmp_save_path)


