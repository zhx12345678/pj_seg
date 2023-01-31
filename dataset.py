import os
import albumentations
import cv2
import numpy as np
import torch
import torch.utils.data
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.geometric.transforms import Flip
from albumentations.augmentations.geometric.rotate import RandomRotate90
from albumentations.augmentations.geometric.resize import Resize
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        pos = torch.tensor(float(img_id.split('_')[3]),dtype = torch.float32)
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext),cv2.IMREAD_GRAYSCALE)

        mask = []

        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = np.expand_dims(img,axis = 2)
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id},pos

if __name__ == '__main__':
    img_ids = [i[0:-4] for i in
               os.listdir('../pj_seg/data/images')]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    train_transform = Compose([
        albumentations.RandomRotate90(),
        albumentations.Flip(),
        albumentations.RandomBrightnessContrast(),
        albumentations.RandomCrop(width=96,height = 96),
        # transforms.Normalize(),
    ])
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('../pj_seg/data/images'),
        mask_dir=os.path.join('../pj_seg/data/masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=4,
        transform=train_transform)
    print(train_dataset[1])


