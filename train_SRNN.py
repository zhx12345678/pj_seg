import argparse
import os
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.geometric.transforms import Flip
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.geometric.resize import Resize
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score, iou_score_multiple_class, dice_score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--data_source', default='../pj_seg/data', type=str,
                        help='data source')
    parser.add_argument('--device',default=None,
                        help='train model device')
    # model
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=240, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=240, type=int,
                        help='image height')

    # dataset
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def train_SRNN(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for a, input, b, c in train_loader:
        input = input.to(device=config['device'])
        # compute output
        output = model(input)
        loss = criterion(output, input)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_meters['loss'].update(loss.item(), input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg)])

def main():
    config = vars(parse_args())
    epochs = 50

    if torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        config['device'] = 'cpu'

    img_ids = [i[0:-4] for i in
               os.listdir(os.path.join(config['data_source'], 'images'))]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        albumentations.PadIfNeeded(min_height=config['input_h'],min_width=config['input_w']),
        RandomRotate90(),
        Flip(),
        albumentations.RandomCrop(config['input_h'], config['input_w']),
        # transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_source'], 'images'),
        mask_dir=os.path.join(config['data_source'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)



    model = archs.__dict__['SRNN'](config['num_classes'])
    model = model.to(device=config['device'])

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)

    params = filter(lambda p: p.requires_grad, model.parameters())

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        train_log = train_SRNN(config, train_loader, model, criterion, optimizer)
        if epoch % 2 == 0 and epoch != 0:
            torch.save(model.state_dict(), 'models/SRNN/model.pth')
            print("model saved")




    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()