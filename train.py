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
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--data_source', default='../pj_seg/data', type=str,
                        help='data source')
    parser.add_argument('--device',default=None,
                        help='train model device')
    parser.add_argument('--SRNN_location', type = str,
                        help = "where the model stored")
    parser.add_argument("--if_pos" , default=False, type = str2bool,
                        help = "if pos")
    parser.add_argument('--random_state',default=41,type = int)
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')

    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=240, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=240, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='Exponential_Logarithmic_Loss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
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


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _, position in train_loader:
        position = torch.reshape(position, (config['batch_size'], 1))
        input = input.to(device=config['device'])
        target = target.to(device=config['device'])
        position = position.to(device=config['device'])

        # compute output
        output, pos = model(input)
        loss = criterion(output, target, pos, position,if_pos = config['if_pos'])
        iou = iou_score_multiple_class(output, target)
        dice = dice_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _, a in val_loader:
            input = input.to(device=config['device'])
            target = target.to(device=config['device'])

            # compute output
            output, pos = model(input)
            loss = criterion(output, target, if_pos=False,if_SRNN = False)
            iou = iou_score_multiple_class(output, target)
            dice = dice_score(output,target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict(
        [('loss', avg_meters['loss'].avg),
         ('iou', avg_meters['iou'].avg),
         ('dice', avg_meters['dice'].avg)]
    )






def main():
    config = vars(parse_args())
    print(config['SRNN_location'])
    if config['name'] is None:
        config['name'] = 'SRSCN'
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    if torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        config['device'] = 'cpu'

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['SRNN_location'] != None:
        SRNN = archs.__dict__['SRNN'](config['num_classes'])
        SRNN_dict = torch.load(config['SRNN_location'])
        SRNN.load_state_dict(SRNN_dict)
    else:
        SRNN = None

    criterion = losses.__dict__[config['loss']](SRNN).to(device=config['device'])

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'], )
    # config['deep_supervision'])

    model = model.to(device=config['device'])

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
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

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    img_ids = [i[0:-4] for i in
               os.listdir(os.path.join(config['data_source'], 'images'))]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['random_state'])

    # train_transform = Compose([
    #     albumentations.RandomRotate90(),
    #     albumentations.Flip(),
    #     albumentations.RandomBrightnessContrast(),
    #     albumentations.RandomCrop(width=config['input_w'],height = config['input_h']),
    #     transforms.Normalize(),
    # ])

    val_transform = Compose([
        albumentations.PadIfNeeded(min_height=config['input_h'], min_width=config['input_w']),
        albumentations.RandomCrop(width=config['input_w'],height = config['input_h']),
        # transforms.Normalize(),
    ])

    train_transform = Compose([
        albumentations.PadIfNeeded(min_height=config['input_h'],min_width=config['input_w']),
        RandomRotate90(),
        Flip(),
        OneOf([
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        albumentations.RandomCrop(config['input_h'], config['input_w']),
        # transforms.Normalize(),
    ])
    #
    # val_transform = Compose([
    #     Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_source'], 'images'),
        mask_dir=os.path.join(config['data_source'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_source'], 'images'),
        mask_dir=os.path.join(config['data_source'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', [])
    ])

    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss {:.4f} - iou {} - val_loss {:.4f} - val_iou {} - val_dice {}'.format(train_log['loss'],
                                                                                         np.around(train_log['iou'], 4),
                                                                                         val_log['loss'],
                                                                                         np.around(val_log['iou'], 4),
                                                                                         np.around(val_log['dice'], 4)))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['dice'][1:].mean() > best_dice:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_dice = val_log['dice'].mean()
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
