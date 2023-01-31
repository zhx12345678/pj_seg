import numpy as np
import torch
import torch.nn.functional as F




def iou_score_multiple_class(output,target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy().astype(np.int8)
    num_class = output.shape[1]
    labels = np.argmax(output,axis = 1)
    iou_score_list = []
    for i in range(num_class):
        labels_oneclass = labels == i
        intersection = (labels_oneclass & target[:,i,:,:]).sum()
        union = (labels_oneclass | target[:,i,:,:]).sum()
        iou_score_list.append((intersection + smooth) / (union + smooth))

    return np.array(iou_score_list)



def dice_score(output,target):
    smooth = 1
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy().astype(np.int8)
    num_class = output.shape[1]
    labels = np.argmax(output,axis = 1)
    dice_score_list = []
    for i in range(num_class):
        labels_oneclass = labels == i
        intersection = (labels_oneclass & target[:,i,:,:]).sum()
        union = labels_oneclass.sum() + target[:,i,:,:].sum()
        dice_score_list.append(2 * intersection / (union + smooth))

    return np.array(dice_score_list)


