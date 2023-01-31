import argparse
import os
from glob import glob
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from metrics import iou_score, iou_score_multiple_class, dice_score
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import archs
from dataset import Dataset
from metrics import iou_score
import albumentations
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="SRSCN_a",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['data_source'] = "/Users/zhouhouxin/PycharmProjects/pythonProject/pj_seg/data"

    if torch.cuda.is_available():
        config['device'] = 'cuda'
    else:
        config['device'] = 'cpu'

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])

    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'], )


    # Data loading code
    img_ids = [i[0:-4] for i in
               os.listdir(os.path.join(config['data_source'], 'images'))]


    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=1)

    model.load_state_dict(torch.load('models/%s/model.pth'%config['name'], map_location = torch.device("cpu")))
    model.eval()

    val_transform = Compose([
        albumentations.PadIfNeeded(min_height=config['input_h'], min_width=config['input_w']),
        albumentations.CenterCrop(width=config['input_w'],height = config['input_h']),
        # transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_source'], 'images'),
        mask_dir=os.path.join(config['data_source'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)


    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    dice = []
    with torch.no_grad():
        for input, target, meta,_ in tqdm(val_loader, total=len(val_loader)):
            input = input.to(config['device'])
            target = target.to(config['device'])

            # compute output
            output = model(input)


            output = output[0].data.cpu().numpy()
            input = input.data.cpu().numpy()
            labels = np.argmax(output, axis=1)

            for i in range(len(output)):
                dice.append(dice_score(output[[i]],target[[i]]))
                image = cv2.cvtColor(input[i][0], cv2.COLOR_BGR2RGB)
                for c in range(config['num_classes']-1):
                    image[:,:,c][(labels==(c+1))[i]] = 1
                cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + 'ini' + '.jpg'),input[i][0] * 255)
                cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),image*255)

    dice = np.stack(dice)
    np.savetxt(config['name'] + ".csv", dice, delimiter=",")
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
