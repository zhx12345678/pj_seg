import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss','Exponential_Logarithmic_Loss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class Exponential_Logarithmic_Loss(nn.Module):
    def __init__(self,SRNN = None):
        super().__init__()
        self.SRNN = SRNN
    def forward(self,input,target,pos = 0,position = 0,gamma_Dice = 0.3,gamma_Cross = 1,w_Dice = 0.8,lambda_SRNN = 5e-4,lambda_SCN = 1e-6,if_pos = True,if_SRNN = True):
        batch_size = target.size(0)
        num_class = target.size(1)
        smooth = 1e-5
        w_Cross = 1 - w_Dice
        input_soft = torch.softmax(input,dim = 1)
        intersection = (input_soft * target)
        dice = (2. * intersection.sum((2,3)) + smooth) / (input_soft.sum((2,3)) + target.sum((2,3)) + smooth)
        L_dice = ((-torch.log(dice))**gamma_Dice).sum()/(batch_size*num_class)
        L_cross = F.cross_entropy(input,target,reduction='sum')/(batch_size*num_class)

        loss = w_Dice * L_dice + w_Cross * L_cross
        if if_pos == True:
            loss += lambda_SCN*nn.functional.mse_loss(pos,position,reduction='sum')
        if self.SRNN != None and if_SRNN == True:
            loss += lambda_SRNN*nn.functional.mse_loss(self.SRNN(input),self.SRNN(target),reduction='sum')
        return loss



