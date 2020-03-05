"""
The code will split the training set into k-fold for cross-validation
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
import shutil

root = '../data/2018/MICCAI_BraTS_2018_Data_Training'
valid_data_dir = '../data/2018/MICCAI_BraTS_2018_Data_Validation'

backup = '../2018/datasets'
backup_files = os.listdir(backup)
if len(backup_files) != 0:
    print("Copy from backup")
    for file in backup_files:
        shutil.copy(os.path.join(backup, file), os.path.join(root, file))
        count=0
        with open(os.path.join(root, file), 'r') as f:
            for line in f:
                count += 1
            print("File {} has {} lines.".format(file, count)
    return None

def write(data, fname, root=root):
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))

limit = float(sys.argv[1])

hgg = os.listdir(os.path.join(root, 'HGG'))
hgg = [os.path.join('HGG', f) for f in hgg]
lgg = os.listdir(os.path.join(root, 'LGG'))
lgg = [os.path.join('LGG', f) for f in lgg]

print("Original size: HGG:{}, LGG:{}, Total:{}".format(len(hgg), len(lgg), len(hgg) + len(lgg)))
hgg = hgg[:int(limit*len(hgg))]
lgg = lgg[:int(limit*len(lgg))]
print("Limited size: HGG:{}, LGG:{}, Total:{}".format(len(hgg), len(lgg), len(hgg) + len(lgg)))
X = hgg + lgg
Y = [1] * len(hgg) + [0] * len(lgg)

write(X, 'all.txt')
shutil.copy(os.path.join(root,'all.txt'), os.path.join(backup, 'all.txt'))
X, Y = np.array(X), np.array(Y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

for k, (train_index, valid_index) in enumerate(skf.split(Y, Y)):
    train_list = list(X[train_index])
    valid_list = list(X[valid_index])

    write(train_list, 'train_{}.txt'.format(k))
    write(valid_list, 'valid_{}.txt'.format(k))

    shutil.copy(os.path.join(root,'train_{}.txt'.format(k)),
                            os.path.join(backup, 'train_{}.txt'.format(k)))
    shutil.copy(os.path.join(root,'valid_{}.txt'.format(k)), 
                            os.path.join(backup, 'valid_{}.txt'.format(k)))

valid = os.listdir(os.path.join(valid_data_dir))
valid = [f for f in valid if not (f.endswith('.csv') or f.endswith('.txt'))]
write(valid, 'valid.txt', root=valid_data_dir)
