"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import cv2
from PIL import Image

import torch
import torch.distributed as dist
import numpy as np
from torchvision import datasets, transforms

from torch.autograd import Variable
import torch.nn as nn

import torch.nn.functional as F
from timm.utils import *
import glob
from models.fan import FANBlock
from timm.data import create_transform


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
            fmt = "{value:.2f} ({global_avg:.2f})"
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
        if isinstance(module, FANBlock):
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


def standard_plot_attn_on_image(original_image, mask, save_path):
    w = int(mask.shape[0]**0.5)
    transformer_attribution = mask.reshape(1,1,w,w)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=224//w, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * transformer_attribution), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image_transformer_attribution)
    cam = cam / np.max(cam)

    vis =  np.uint8(255 * cam)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    # Image.fromarray(vis).save(save_path, quality=85, optimize=True)
    out = Image.fromarray(vis)
    t = transforms.ToTensor()
    out = t(out)
    return out


def _threshold(x, threshold=None):

    if threshold is not None:
        #         return (x > threshold).type(x.dtype)
        quantile = torch.quantile(x.to(torch.float), threshold, dim=-1).unsqueeze(-1)
        return (x > quantile).type(x.dtype)
    else:
        return x


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.min = 1000000
        self.max = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


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


