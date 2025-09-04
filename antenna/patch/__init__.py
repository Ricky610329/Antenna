from ..utils import *
from .patch_simulator import  com_error
from .patch_simulator.dual_port import DualPortSimulator
from .patch_simulator.single_port import SinglePortSimulator

import torch

def custom_loss_r(prediciton, target, loss_type='SmoothL1Loss'):
    
    criterion_r = nn.SmoothL1Loss() if loss_type=='SmoothL1Loss' else nn.MSELoss()

    high_response = target.max()
    low_response = target.min()
    
    mask_25 = target == high_response  # mask == -2.5 index
    mask_b_25 = prediciton[mask_25] < high_response
    
    if mask_b_25.sum()==0:
        loss_25 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_25 = criterion_r(prediciton[mask_25][mask_b_25], target[mask_25][mask_b_25])
        
    mask_10 = target == low_response  # mask == -2.5 index
    mask_b_10 = prediciton[mask_10] > low_response
    
    if mask_b_10.sum()==0:
        loss_10 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_10 = criterion_r(prediciton[mask_10][mask_b_10], target[mask_10][mask_b_10])
    
    loss = loss_25 + loss_10
    return loss

def custom_loss_g(prediciton, target, loss_type='SmoothL1Loss'):
    
    criterion_g = nn.SmoothL1Loss() if loss_type=='SmoothL1Loss' else nn.MSELoss()

    high_gain = target.max()
    low_gain = target.min()
    
    mask_10 = target == low_gain  # mask == -10 index
    mask_b_10 = prediciton[mask_10] > low_gain
    
    if mask_b_10.sum()==0:
        loss_10 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_10 = criterion_g(prediciton[mask_10][mask_b_10], target[mask_10][mask_b_10])
        
    mask_4 = target == high_gain  # mask == 4 index
    mask_b_4 = prediciton[mask_4] < high_gain
    
    if mask_b_4.sum()==0:
        loss_4 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_4 = criterion_g(prediciton[mask_4][mask_b_4], target[mask_4][mask_b_4])
    
    loss = loss_10 + loss_4
    return loss