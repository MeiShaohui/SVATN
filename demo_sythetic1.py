# -*- coding: utf-8 -*-
"""
MAIN FUNCTION
"""

from __future__ import print_function
#from .common_utils import *

from abc import ABC
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import scipy.io
from tqdm import tqdm
from numpy import linalg as LA

import torch.optim
import torch.nn as nn
from models import *
from utils import *

EVALUATION = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


# MODEL
class CAE_AbEst(nn.Module, ABC):
    def __init__(self):
        super(CAE_AbEst, self).__init__()
        self.conv1 = nn.Sequential(
            UnmixArch(
                input_depth,
                EE.shape[1],
                # num_channels_down = [8, 16, 32, 64, 128],
                # num_channels_up   = [8, 16, 32, 64, 128],
                # num_channels_skip = [4, 4, 4, 4, 4],
                num_channels_down=[256],
                num_channels_up=[256],
                num_channels_skip=[4],
                filter_size_up=3,
                filter_size_down=3,
                filter_skip_size=1,
                upsample_mode='bilinear',  # downsample_mode='avg',
                need1x1_up=True,
                need_sigmoid=True,
                need_bias=True,
                pad=pad,
                act_fun='LeakyReLU').type(dtype))

    def forward(self, x):
        x = self.conv1(x)
        return x

# LOSS
def my_loss(target, lamb, out_1, out_2):
    loss1 = 0.5*torch.norm((out_1.transpose(1,0).view(1,p1,nr1,nc1) - target), 'fro')**2
    loss2 = 0.5*torch.norm((out_1.transpose(1,0).view(1,p1,nr1,nc1) + out_2.transpose(1,0).view(1,p1,nr1,nc1) - target), 'fro')**2
    return loss1+lamb*loss2

def closure1():
    global i, out_LR_np, out_avg_np, out_LR, out_HR \
        , out_SV_LR_np, out_SV_avg_np, out_SV, out_SV_avg, out_SV_HR, out_SV_HR_avg \
        , out_avg, out_HR_np, out_HR_avg, out_HR_avg_np, RMSE_LR_last, last_net \
        , net_input, loss
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_LR = net1(net_input1)
    out_HR = torch.mm(E_torch.view(p1, LibS),
                      out_LR.view(LibS, nr1 * nc1))
    out_SV_LR = net2(net_input1)
    out_SV_HR = torch.mm(SV_torch.view(p1, LibS),
                         out_SV_LR.view(LibS, nr1 * nc1))
    # Smoothing
    if out_avg is None:
        out_avg = out_LR.detach()
        out_HR_avg = out_HR.detach()
        out_SV_avg = out_SV_LR.detach()
        out_SV_HR_avg = out_SV_HR.detach()
    else:
        out_avg = out_avg * exp_weight + out_LR.detach() * (1 -
                                                            exp_weight)
        out_HR_avg = out_HR_avg * exp_weight + out_HR.detach() * (
                1 - exp_weight)
        out_SV_avg = out_SV_avg * exp_weight + out_SV_LR.detach() * (1 -
                                                                     exp_weight)
        out_SV_HR_avg = out_SV_HR_avg * exp_weight + out_SV_HR.detach() * (
                1 - exp_weight)

    out_HR = out_HR.view((1, p1, nr1, nc1))
    out_SV_HR = out_SV_HR.view((1, p1, nr1, nc1))
    total_loss = my_loss(img_noisy_torch, 1, out_HR, out_SV_HR)
    total_loss.backward()
    loss[i] = total_loss.item()
    i += 1

    return total_loss, loss


#  LOAD DATA
image_file = "data/data_sythetic1.mat"
data = scipy.io.loadmat(image_file)
img_np_gt = data["Y_clean"].transpose(2, 0, 1)
A_true_np = data['XT']
EE = data["EE"]
SV = data["SV"]
[p1, nr1, nc1] = img_np_gt.shape
LibS = EE.shape[1]

npar = np.zeros((1, 3))
npar[0, 0] = 13.3
npar[0, 1] = 41.4
npar[0, 2] = 130.8

tol1 = npar.shape[1]
tol2 = 1
save_result = False

rmax = 5

for fi in tqdm(range(tol1)):
    for fj in tqdm(range(tol2)):
        img_noisy_np = add_noise(
            img_np_gt, 1 / npar[0, fi])  
        print('signal-to-noise:', compare_snr(img_np_gt, img_noisy_np))        
        img_resh = np.reshape(img_noisy_np, (p1, nr1 * nc1))
        V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
        PC = np.diag(SS) @ U
        img_resh_DN = V[:, :rmax] @ PC[:rmax, :]
        img_noisy_np = np.reshape(np.clip(img_resh_DN, 0, 1), (p1, nr1, nc1))
        INPUT = 'noise'  
        pad = 'reflection'
        need_bias = True
        OPT_OVER = 'net'  
        
        reg_noise_std = 0.0
        LR1 = 0.01

        OPTIMIZER1 = 'adam'  # 'RMSprop'#'adam' # 'LBFGS'
        show_every = 3000
        exp_weight = 0.99
        if fi == 0:
            num_iter1 = 1000
        elif fi == 1:
            num_iter1 = 2000
        elif fi == 2:
            num_iter1 = 4000
        input_depth = img_noisy_np.shape[0]

        net1 = CAE_AbEst()
        net1.cuda()
        net2 = CAE_AbEst()
        net2.cuda()

        img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
        net_input1 = get_noise(input_depth, INPUT,
                               (img_noisy_np.shape[1],
                                img_noisy_np.shape[2])).type(dtype).detach()
        #net_input1 = img_noisy_torch
        E_torch = np_to_torch(EE).type(dtype)
        SV_torch = np_to_torch(SV).type(dtype)
      
        net_input_saved = net_input1.detach().clone()
        noise = net_input1.detach().clone()
        out_avg = None
        out_HR_avg = None
        out_SV_avg = None
        out_SV_HR_avg = None        
        last_net = None
        RMSE_LR_last = 0
        loss = np.zeros((num_iter1, 1))
        i = 0

        p11 = get_params(OPT_OVER, net1, net_input1)
        p12 = get_params(OPT_OVER, net2, net_input1)
        parameterall = p11 + p12

        torch.cuda.synchronize()
        time_start = time.time()

        optimize(OPTIMIZER1, parameterall, closure1, LR1, num_iter1)

        torch.cuda.synchronize()
        time_end = time.time()

        if EVALUATION:
            out_avg_np = out_avg.detach().cpu().squeeze().numpy()
            out_SV_HR_np = out_SV_HR.detach().cpu().squeeze().numpy()
            out_HR_np = out_HR.detach().cpu().squeeze().numpy()

            SRE_avg = 10 * np.log10(
                LA.norm(
                    A_true_np.astype(np.float32).reshape(
                        (EE.shape[1], nr1 * nc1)), 'fro') / LA.norm(
                            (A_true_np.astype(np.float32) -
                             np.clip(out_avg_np, 0, 1)).reshape(
                                 (EE.shape[1], nr1 * nc1)), 'fro'))
            SRE_SV = 10 * np.log10(
                LA.norm(
                    img_noisy_np.astype(np.float32).reshape(
                        (p1, nr1 * nc1)), 'fro') / LA.norm(
                            (img_noisy_np.astype(np.float32)-
                            np.clip(out_HR_np, 0, 1)-np.clip(out_SV_HR_np, 0, 1)).reshape(
                            (p1, nr1 * nc1)), 'fro'))                                 

            print('**********************************')
            print('SRE_A: {:.5f}'.format(SRE_avg))
            print('SRE_R: {:.5f}'.format(SRE_SV))
            print('total computational cost: {:.5f}'.format(time_end-time_start))
            print('**********************************')
