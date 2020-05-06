"""
Load the 'nii' file and save as pkl file.
Carefully check your path please.
"""
import time
import os
import pickle
import sys
import nibabel as nib
import numpy as np
import argparse
from utils import Parser
from functools import partial
from torchvision import transforms
from utils.utils import *

dist_map_transform = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=4),
        itemgetter(0),
        lambda t: t.cpu().numpy(),
        one_hot2dist,
        # lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])


args = Parser()
modalities = ('flair', 't1ce', 't1', 't2')

train_set = {
    'root': '../data/2018/MICCAI_BraTS_2018_Data_Training',
    'flist': 'all.txt',
}

valid_set = {
    'root': '../data/2018/MICCAI_BraTS_2018_Data_Validation',
    'flist': 'valid.txt',
}

test_set = {
    'root': '../data/2018/MICCAI_BraTS_2018_Data_TTest',
    'flist': 'test.txt',
}


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def normalize(image, mask=None):
    assert len(image.shape) == 3  # shape is [H,W,D]
    assert image[0, 0, 0] == 0  # check the background is zero
    if mask is not None:
        mask = (image > 0)  # The bg is zero

    mean = image[mask].mean()
    std = image[mask].std()
    image = image.astype(dtype=np.float32)
    image[mask] = (image[mask] - mean) / std
    return image


def savepkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def write(data, fname, root):
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))

def process_f32(path, save=True):
    """ Set all Voxels that are outside of the brain mask to 0"""
    start = time.time()
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')
    if os.path.exists(path + 'dist_maps.nii.gz'):
        dist_maps = np.array(nib_load(path + 'dist_maps.nii.gz'), dtype='float32', order='C')
    else:
        if label != np.array([1]):    
            dist_maps = dist_map_transform(label)
            nib.save(nib.Nifti1Image(dist_maps, None), path + 'dist_maps.nii.gz')
        else:
            dist_maps = np.array([1])
    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C')
        for modal in modalities], -1)

    mask = images.sum(-1) > 0

    for k in range(4):
        x = images[..., k]  #
        y = x[mask]  #

        lower = np.percentile(y, 0.2)  # 算分位数
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]

        x -= y.mean()
        x /= y.std()

        images[..., k] = x
    if save:
        output = path + 'data_f32.pkl'
        savepkl(data=(images, label), path=output)
        print("It takes {:.2f}s to save:{}".format(time.time() -start, output))

    return images, label, dist_maps
    # return images, label


def doit(dset, limit=1, change_size=False):
    root = dset['root']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    
    print("Total samples number:", len(subjects))
    subjects = subjects[:int(limit*len(subjects))]
    print("Limited samples number:", len(subjects))
    if change_size:
        write(subjects, dset['flist'], root)
    else:
        print("No change in file list")
    names = [sub.split('/')[-1] for sub in subjects]
    paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]
    for path in paths:
        process_f32(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-limit', '--limit', default=1, type=float,
                    help='Limit rate')
    parser.add_argument('-change_size', '--change_size', default=False, type=bool,
                    help='Save new dataset size')
    ## parse arguments
    args = parser.parse_args()
    doit(train_set, limit=args.limit, change_size=args.change_size)
    # doit(valid_set, limit=limit)
    # doit(test_set)
