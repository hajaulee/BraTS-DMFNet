# coding=utf-8
import sys
import argparse
import logging
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

import numpy as np
import setproctitle
import models
from data import datasets
from data.sampler import CycleSampler
from data.data_utils import init_fn
from utils import Parser, criterions

from predict import set_predict_device, validate_softmax, AverageMeter

# pip install setproctitle
cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--cfg', default='3DUNet_dice_fold0', required=True, type=str,
                    help='Your detailed configuration of the network')
parser.add_argument('-gpu', '--gpu', default='0', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('-restore', '--restore', default='model_last.pth', type=str)  # model_last.pth
parser.add_argument('-output_path', '--output_path', default='ckpts', type=str)
parser.add_argument('-prefix_path', '--prefix_path', default='', type=str)

path = os.path.dirname(__file__)

args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)

ckpts = args.makedir()
args.resume = os.path.join(ckpts, args.restore)  # specify the epoch
cuda_ids = [int(i) for i in args.gpu.split(',')]
print(args)
set_predict_device(args.gpu)
def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print("Cuda number:", torch.cuda.device_count())
    device = torch.device("cuda:" + ','.join(map(str, cuda_ids)) if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    Network = getattr(models, args.net)  #
    model = Network(**args.net_params)
    # model = torch.nn.DataParallel(model).to(device)
    model = torch.nn.DataParallel(model, device_ids=cuda_ids)  # .cuda()
    optimizer = getattr(torch.optim, args.opt)(model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)

    msg = ''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            msg = ("=> loaded checkpoint '{}' (iter {})"
                   .format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
    else:
        msg = '-------------- New training session ----------------'

    msg += '\n' + str(args)
    logging.info(msg)

    Dataset = getattr(datasets, args.dataset)  #

    if args.prefix_path:
        args.train_data_dir = os.path.join(args.prefix_path, args.train_data_dir)
    train_list = os.path.join(args.train_data_dir, args.train_list)
    train_set = Dataset(train_list, root=args.train_data_dir, for_train=True,
                        transforms=args.train_transforms)

    num_iters = args.num_iters or (len(train_set) * args.num_epochs) // args.batch_size
    num_iters -= args.start_iter
    train_sampler = CycleSampler(len(train_set), num_iters * args.batch_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collate, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, worker_init_fn=init_fn)

    if args.valid_list:
        valid_list = os.path.join(args.train_data_dir, args.valid_list)
        valid_set = Dataset(valid_list,
                            root=args.train_data_dir,
                            for_train=False,
                            transforms=args.test_transforms)

        valid_loader = DataLoader(
            valid_set,
            batch_size=1,
            shuffle=False,
            collate_fn=valid_set.collate,
            num_workers=args.workers,
            pin_memory=True)

    start = time.time()

    enum_batches = len(train_set) / float(args.batch_size)  # nums_batch per epoch
    args.schedule = {int(k * enum_batches): v for k, v in args.schedule.items()}  # 17100
    # args.save_freq = int(enum_batches * args.save_freq)
    # args.valid_freq = int(enum_batches * args.valid_freq)

    losses = AverageMeter()
    torch.set_grad_enabled(True)

    start_epoch = time.time()
    elapsed_bsize = 0
    for i, data in enumerate(train_loader, args.start_iter):
        if int(i / enum_batches) + 1 != elapsed_bsize:
            print("New epoch: ", elapsed_bsize + 1)
            print("Time cost: {:.2f}s/epoch".format(time.time() - start_epoch))
            start_epoch = time.time()
        elapsed_bsize = int(i / enum_batches) + 1
        epoch = int((i + 1) / enum_batches)
        setproctitle.setproctitle("Epoch:{}/{}".format(elapsed_bsize, args.num_epochs))

        adjust_learning_rate(optimizer, epoch, args.num_epochs, args.opt_params.lr)

        # data = [t.cuda(non_blocking=True) for t in data]
        data = [t.to(device) for t in data]
        x, target = data[:2]
        # print(x.shape, target.shape)
        output = model(x)
        if not args.weight_type:  # compatible for the old version
            args.weight_type = 'square'

        # loss = criterion(output, target, args.eps,args.weight_type)
        # loss = criterion(output, target,args.alpha,args.gamma) # for focal loss
        loss = criterion(output, target, *args.kwargs)

        # measure accuracy and record loss
        losses.update(loss.item(), target.numel())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % int(enum_batches * args.save_freq) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 1)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 2)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 3)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 4)) == 0:
            file_name = os.path.join(ckpts, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'iter': i + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)
            
            file_name = os.path.join(ckpts, 'model_last.pth')
            torch.save({
                'iter': i,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)
        # validation
        if (i + 1) % int(enum_batches * args.valid_freq) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 1)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 2)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 3)) == 0 \
                or (i + 1) % int(enum_batches * (args.num_epochs - 4)) == 0:
            logging.info('-' * 50)
            msg = 'Iter {}, Epoch {:.4f}, {}'.format(i, i / enum_batches, 'validation')
            logging.info(msg)
            with torch.no_grad():
                validate_softmax(
                    valid_loader,
                    model,
                    cfg=args.cfg,
                    savepath='',
                    names=valid_set.names,
                    scoring=True,
                    verbose=False,
                    use_TTA=False,
                    snapshot=False,
                    postprocess=False,
                    cpu_only=False,
                    epoch=i // enum_batches)
                # print(scores)
        msg = 'Iter {0:}, Epoch {1:.4f}, Loss {2:.7f}'.format(
            i + 1, (i + 1) / enum_batches, losses.avg)

        logging.info(msg)
        losses.reset()

    i = num_iters + args.start_iter
    file_name = os.path.join(ckpts, 'model_last.pth')
    torch.save({
        'iter': i,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        file_name)

    msg = 'total time: {:.4f} minutes'.format((time.time() - start) / 60)
    logging.info(msg)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


if __name__ == '__main__':
    main()
