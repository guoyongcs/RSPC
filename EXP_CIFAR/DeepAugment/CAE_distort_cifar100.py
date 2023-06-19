import sys
import os
import numpy as np
import os
import shutil
import tempfile
from PIL import Image
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as trn
from torchvision import datasets
import torchvision.transforms.functional as trnF 
from torch.nn.functional import gelu, conv2d
import torch.nn.functional as F
import random
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from CAE_Model.cae_32x32x32_zero_pad_bin import CAE
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


test_transform = trn.Compose([trn.Resize(512), trn.ToTensor()])

def get_weights():
    weight_keys = ['e_conv_1.1.weight', 'e_conv_1.1.bias', 'e_conv_2.1.weight', 'e_conv_2.1.bias', 'e_block_1.1.weight', 'e_block_1.1.bias', 'e_block_1.4.weight', 'e_block_1.4.bias', 'e_block_2.1.weight', 'e_block_2.1.bias', 'e_block_2.4.weight', 'e_block_2.4.bias', 'e_block_3.1.weight', 'e_block_3.1.bias', 'e_block_3.4.weight', 'e_block_3.4.bias', 'e_conv_3.0.weight', 'e_conv_3.0.bias', 'd_up_conv_1.0.weight', 'd_up_conv_1.0.bias', 'd_up_conv_1.3.weight', 'd_up_conv_1.3.bias', 'd_block_1.1.weight', 'd_block_1.1.bias', 'd_block_1.4.weight', 'd_block_1.4.bias', 'd_block_2.1.weight', 'd_block_2.1.bias', 'd_block_2.4.weight', 'd_block_2.4.bias', 'd_block_3.1.weight', 'd_block_3.1.bias', 'd_block_3.4.weight', 'd_block_3.4.bias', 'd_up_conv_2.0.weight', 'd_up_conv_2.0.bias', 'd_up_conv_2.3.weight', 'd_up_conv_2.3.bias', 'd_up_conv_3.0.weight', 'd_up_conv_3.0.bias', 'd_up_conv_3.3.weight', 'd_up_conv_3.3.bias']
    key_mapping = dict([(str(int(i / 2)) + ".weight", key) if i % 2 == 0 else (str(int(i / 2)) + ".bias", key) for i, key in enumerate(weight_keys)])
    NUM_LAYERS = int(len(key_mapping.values()) / 2) # 21
    NUM_DISTORTIONS = 8
    MODEL_PATH = "CAE_Weights/model_final.state"
    OPTION_LAYER_MAPPING = {0: range(11, NUM_LAYERS - 5), 1: range(8, NUM_LAYERS - 7), 2: range(8, NUM_LAYERS - 7), 3: range(10, NUM_LAYERS - 7), 4: range(8, NUM_LAYERS - 7), 5: range(8, NUM_LAYERS - 7), 6: range(8, NUM_LAYERS - 7), 7: range(8, NUM_LAYERS - 7), 8: range(8, NUM_LAYERS - 7)}

    def get_name(i, tpe):
        return key_mapping[str(i) + "." + tpe]

    weights = torch.load(MODEL_PATH)
    for option in random.sample(range(NUM_DISTORTIONS), 1):
        i = np.random.choice(OPTION_LAYER_MAPPING[option])
        j = np.random.choice(OPTION_LAYER_MAPPING[option])
        weight_i = get_name(i, "weight")
        bias_i = get_name(i, "bias")
        weight_j = get_name(j, "weight")
        bias_j = get_name(j, "weight")
        if option == 0:
            weights[weight_i] = torch.flip(weights[weight_i], (0,))
            weights[bias_i] = torch.flip(weights[bias_i], (0,))
            weights[weight_j] = torch.flip(weights[weight_j], (0,))
            weights[bias_j] = torch.flip(weights[bias_j], (0,))
        elif option == 1:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(12)]:
                weights[weight_i][k] = -weights[weight_i][k]
                weights[bias_i][k] = -weights[bias_i][k]
        elif option == 2:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(25)]:
                weights[weight_i][k] = 0 * weights[weight_i][k]
                weights[bias_i][k] = 0 * weights[bias_i][k]
        elif option == 3:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(25)]:
                weights[weight_i][k] = -gelu(weights[weight_i][k])
                weights[bias_i][k] = -gelu(weights[bias_i][k])
        elif option == 4:
            weights[weight_i] = weights[weight_i] *\
            (1 + 2 * np.float32(np.random.uniform()) * (4*torch.rand_like(weights[weight_i]-1)))
            weights[weight_j] = weights[weight_j] *\
            (1 + 2 * np.float32(np.random.uniform()) * (4*torch.rand_like(weights[weight_j]-1)))
        elif option == 5: ##### begin saurav #####
            if random.random() < 0.5:
                mask = torch.round(torch.rand_like(weights[weight_i]))
            else:
                mask = torch.round(torch.rand_like(weights[weight_i])) * 2 - 1
            weights[weight_i] *= mask
        elif option == 6:
            _k = random.randint(1, 3)
            weights[weight_i] = torch.rot90(weights[weight_i], k=_k, dims=(2,3))
        elif option == 7:
            out_filters = weights[weight_i].shape[0]
            to_zero = list(set([random.choice(list(range(out_filters))) for _ in range(out_filters // 5)]))
            weights[weight_i][to_zero] = weights[weight_i][to_zero] * -1.0
        elif option == 8:
            # Only keep the max filter value in the conv 
            c1, c2, width = weights[weight_i].shape[0], weights[weight_i].shape[1], weights[weight_i].shape[2]
            assert weights[weight_i].shape[2] == weights[weight_i].shape[3]

            w = torch.reshape(weights[weight_i], shape=(c1, c2, width ** 2))
            res = torch.topk(w, k=1)

            w_new = torch.zeros_like(w).scatter(2, res.indices, res.values)
            w_new = w_new.reshape(c1, c2, width, width)
            weights[weight_i] = w_new
        
    return weights    

net = CAE()
net.load_state_dict(get_weights())
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

        save_path = '../../data/CIFAR-100-DeepAugment/CAE/' + self.idx_to_class[target]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path = os.path.join(save_path, str(index) + '.jpg')

        if np.random.uniform() < 0.05:
            weights = get_weights()
            net.load_state_dict(weights)
            net.eval()

        with torch.no_grad():
            img = trnF.to_pil_image(net(sample.unsqueeze(0).cuda()).squeeze().to('cpu').clamp(0, 1))

        img.save(save_path)

        return 0

distorted_dataset = FolderWithPath(
    root="../../data", transform=test_transform)

loader = torch.utils.data.DataLoader(distorted_dataset, batch_size=1, shuffle=True)

for _ in tqdm(loader): 
    continue



