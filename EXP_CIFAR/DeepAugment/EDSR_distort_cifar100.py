import torch
from EDSR_Model import common
import torch.nn as nn
from torchvision import transforms as trn
from torchvision import datasets
import torchvision.transforms.functional as trnF
import numpy as np
from torch.nn.functional import gelu
from torch.nn.functional import conv2d
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import sys
import os
from PIL import Image

import torch.nn.functional as F

from tqdm import tqdm

import os
import shutil
import tempfile
import random

import torchvision
import torchvision.datasets as dset

import argparse
parser = argparse.ArgumentParser(description='Fine-tune')
parser.add_argument('--total-workers', default=None, type=int, required=True)
parser.add_argument('--worker-number', default=None, type=int, required=True) # MUST BE 0-INDEXED
args = parser.parse_args()

all_classes = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
               '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
               '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
               '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
               '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
               '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',
               '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
               '90', '91', '92', '93', '94', '95', '96', '97', '98', '99',]
assert len(all_classes) == 100

# Subset for this worker
classes_chosen = np.array_split(all_classes, args.total_workers)[args.worker_number]

class ImageNetSubsetDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class to take a specified subset of some larger dataset
    """
    def __init__(self, root, *args, **kwargs):

        print("Using {0} classes {1}".format(len(classes_chosen), classes_chosen))

        self.new_root = tempfile.mkdtemp()
        for _class in classes_chosen:
            orig_dir = os.path.join(root, _class)
            assert os.path.isdir(orig_dir)

            os.symlink(orig_dir, os.path.join(self.new_root, _class))

        super().__init__(self.new_root, *args, **kwargs)

        return self.new_root

    def __del__(self):
        # Clean up
        shutil.rmtree(self.new_root)


# test_transform = lambda x: trnF.to_tensor(trnF.resize(x, 128))
test_transform = trn.Compose([trn.Resize(128), trn.ToTensor()])


def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self,
                 n_resblocks=16, n_feats=64, scale=4, res_scale=1, rgb_range=255, n_colors=3,
                 conv=common.default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, pre_distortions={None}, body_distortions={None}):
        # print("Using FF pre distortions = ", pre_distortions)
        # print("Using FF body distortions = ", body_distortions)
        x = self.sub_mean(x)
        x = self.head(x)

        ######################################################################
        # PRE - DISTORTIONS
        ######################################################################

        if 1 in pre_distortions:
            for _ in range(5):
                c1, c2 = random.randint(0, 63), random.randint(0, 63)
                x[:, c1], x[:, c2] = x[:, c2], x[:, c1]

        if 2 in pre_distortions:
            rand_filter_weight = torch.round(torch.rand_like(x) + 0.45) # Random matrix of 1s and 0s
            x = x * rand_filter_weight

        if 3 in pre_distortions:
            rand_filter_weight = (torch.round(torch.rand_like(x) + 0.475) * 2) - 1 # Random matrix of 1s and -1s
            x = x * rand_filter_weight

        ######################################################################
        # BODY - DISTORTIONS
        ######################################################################

        if 1 in body_distortions:
            res = self.body[:5](x)
            res = -res
            res = self.body[5:](res)
        elif 2 in body_distortions:
            if random.randint(0, 2) == 1:
                act = F.relu
            else:
                act = F.gelu
            res = self.body[:5](x)
            res = act(res)
            res = self.body[5:](res)
        elif 3 in body_distortions:
            if random.randint(0, 2) == 1:
                axes = [1, 2]
            else:
                axes = [1, 3]
            res = self.body[:5](x)
            res = torch.flip(res, axes)
            res = self.body[5:](res)
        elif 4 in body_distortions:
            to_skip = set([random.randint(2, 16) for _ in range(3)])
            for i in range(len(self.body)):
                if i not in to_skip:
                    res = self.body[i](x)
        else:
            res = self.body(x)

        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

def get_weights():
    weights = torch.load('./EDSR_Weights/edsr_baseline_x4.pt')

    random_sample_list = np.random.randint(0,17, size=3)
    for option in list(random_sample_list):
        if option == 0:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = torch.flip(weights['body.'+str(i)+'.body.0.weight'], (0,))
            weights['body.'+str(i)+'.body.0.bias'] = torch.flip(weights['body.'+str(i)+'.body.0.bias'], (0,))
            weights['body.'+str(i)+'.body.2.weight'] = torch.flip(weights['body.'+str(i)+'.body.2.weight'], (0,))
            weights['body.'+str(i)+'.body.2.bias'] = torch.flip(weights['body.'+str(i)+'.body.2.bias'], (0,))
        elif option == 1:
            i = np.random.choice(np.arange(1,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = -weights['body.'+str(i)+'.body.0.weight']
            weights['body.'+str(i)+'.body.0.bias'] = -weights['body.'+str(i)+'.body.0.bias']
        elif option == 2:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = 0 * weights['body.'+str(i)+'.body.0.weight']
            weights['body.'+str(i)+'.body.0.bias'] = 0*weights['body.'+str(i)+'.body.0.bias']
        elif option == 3:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = -gelu(weights['body.'+str(i)+'.body.0.weight'])
            weights['body.'+str(i)+'.body.2.weight'] = -gelu(weights['body.'+str(i)+'.body.2.weight'])
        elif option == 4:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = weights['body.'+str(i)+'.body.0.weight'] *\
            torch.Tensor([[0, 1, 0],[1, -4., 1], [0, 1, 0]]).view(1,1,3,3).cuda()
        elif option == 5:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = weights['body.'+str(i)+'.body.0.weight'] *\
            torch.Tensor([[-1, -1, -1],[-1, 8., -1], [-1, -1, -1]]).view(1,1,3,3).cuda()
        elif option == 6:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.2.weight'] = weights['body.'+str(i)+'.body.2.weight'] *\
            (1 + 2 * np.float32(np.random.uniform()) * (2*torch.rand_like(weights['body.'+str(i)+'.body.2.weight']-1)))
        elif option == 7:
            i = np.random.choice(np.arange(0,10,3))
            weights['body.'+str(i)+'.body.0.weight'] = torch.flip(weights['body.'+str(i)+'.body.0.weight'], (-1,))
            weights['body.'+str(i)+'.body.2.weight'] = -1 * weights['body.'+str(i)+'.body.2.weight']
        elif option == 8:
            i = np.random.choice(np.arange(1,13,4))
            z = torch.zeros_like(weights['body.'+str(i)+'.body.0.weight'])
            for j in range(z.size(0)):
                shift_x, shift_y = np.random.randint(3, size=(2,))
                z[:,j,shift_x,shift_y] = np.random.choice([1.,-1.])
            weights['body.'+str(i)+'.body.0.weight'] = conv2d(weights['body.'+str(i)+'.body.0.weight'], z, padding=1)
        elif option == 9:
            i = np.random.choice(np.arange(0,10,3))
            z = (2*torch.rand_like(weights['body.'+str(i)+'.body.0.weight'])*np.float32(np.random.uniform()) - 1)/6.
            weights['body.'+str(i)+'.body.0.weight'] = conv2d(weights['body.'+str(i)+'.body.0.weight'], z, padding=1)
        elif option == 10:
            i = np.random.choice(np.arange(1,12,4))
            z = torch.FloatTensor(np.random.dirichlet([0.1] * 9, (64,64))).view(64,64,3,3).cuda() # 2.weight
            weights['body.'+str(i)+'.body.2.weight'] = conv2d(weights['body.'+str(i)+'.body.2.weight'], z, padding=1)
        elif option == 11: ############ Start Saurav's changes ############
            i = random.choice(list(range(15)))
            noise = (torch.rand_like(weights['body.'+str(i)+'.body.2.weight']) - 0.5) * 1.0
            weights['body.'+str(i)+'.body.2.weight'] += noise
        elif option == 12:
            _ij = [[random.choice(list(range(15))), random.choice([0, 2])] for _ in range(5)]
            for i, j in _ij:
                _k = random.randint(1, 3)
                if random.randint(0, 1) == 0:
                    _dims = (2,3)
                else:
                    _dims = (0,1)
                weights['body.'+str(i)+'.body.'+str(j)+'.weight'] = torch.rot90(weights['body.'+str(i)+'.body.'+str(j)+'.weight'], k=_k, dims=_dims)
        elif option == 13:
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                rand_filter_weight = torch.round(torch.rand_like(weights['body.'+str(i)+'.body.0.weight'])) * 2 - 1 # Random matrix of 1s and -1s
                weights['body.'+str(i)+'.body.0.weight'] = weights['body.'+str(i)+'.body.0.weight'] * rand_filter_weight
        elif option == 14:
            # Barely noticable difference here
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                rand_filter_weight = torch.round(torch.rand_like(weights['body.'+str(i)+'.body.0.weight'])) # Random matrix of 1s and 0s
                weights['body.'+str(i)+'.body.0.weight'] = weights['body.'+str(i)+'.body.0.weight'] * rand_filter_weight
        elif option == 15:
            # Negate some entire filters. Definitely a noticable difference
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                filters_to_be_zeroed = [random.choice(list(range(64))) for _ in range(32)]
                weights['body.'+str(i)+'.body.0.weight'][filters_to_be_zeroed] *= -1
        elif option == 16:
            # Only keep the max filter value in the conv
            _ij = [[random.choice(list(range(15))), random.choice([0, 2])] for _ in range(5)]
            for i, j in _ij:
                w = torch.reshape(weights['body.'+str(i)+'.body.'+str(j)+'.weight'], shape=(64, 64, 9))
                res = torch.topk(w, k=1)

                w_new = torch.zeros_like(w).scatter(2, res.indices, res.values)
                w_new = w_new.reshape(64, 64, 3, 3)
                weights['body.'+str(i)+'.body.'+str(j)+'.weight'] = w_new
        else:
            raise NotImplementedError()
    return weights

weights = get_weights()

net = EDSR()
net.load_state_dict(weights)
net.cuda()
net.eval()


def find_classes(dir):
    classes = all_classes
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class FolderWithPath(dset.CIFAR100):
    def __init__(self, root, transform, **kwargs):
        new_root = super(FolderWithPath, self).__init__(root, transform=transform)

        classes, class_to_idx = find_classes(new_root)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.data[index]
        target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        # save_path = '~/data/hendrycks/DistortedImageNet/' + str(self.option) + '/' + self.idx_to_class[target]
        save_path = '../../data/CIFAR-100-DeepAugment/EDSR/' + self.idx_to_class[target]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(save_path, str(index) + '.jpg')

        if np.random.uniform() < 0.05:
            weights = get_weights()
            net.load_state_dict(weights)
        with torch.no_grad():
            pre_dist = set([random.randint(1, 4) for _ in range(1)])
            body_dist = set([random.randint(1, 5)])
            img = trnF.to_pil_image(net(255*sample.unsqueeze(0).cuda(), pre_distortions=pre_dist, body_distortions=body_dist).squeeze().to('cpu').clamp(0, 255)/255.)

        img.save(save_path)

        return 0


distorted_dataset = FolderWithPath(
    root="../../data", transform=test_transform)

# distorted_dataset[0]

loader = torch.utils.data.DataLoader(
  distorted_dataset, batch_size=16, shuffle=True)

for _ in tqdm(loader): continue
