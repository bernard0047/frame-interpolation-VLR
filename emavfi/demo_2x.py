import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
import os
'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from utils import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small', type=str)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()


print(f'=========================Start Generating=========================')

I0 = cv2.imread('/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/data/dataset/apple/110_13051_23361_12/frame000001.jpg')
I2 = cv2.imread('/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/data/dataset/apple/110_13051_23361_12/frame000019.jpg') 

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

mid = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
os.makedirs('compare/12_img',exist_ok=True)
mimsave('compare/12_img/apple_2x_baseline_ours_small.gif', images, fps=3)
for i in range(len(images)):
    img = cv2.cvtColor(images[i],cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"compare/12_img/img_{i}.png",img)
   


print(f'=========================Done=========================')