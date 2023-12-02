from utils import InputPadder
from Trainer import Model
import config as cfg
import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small_t', type=str)
parser.add_argument('--n', default=16, type=int)
args = parser.parse_args()
assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small_t':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 4, 4]
    )
model = Model(-1,use_perceptual_loss= False)
model.load_model()
model.eval()
model.get_device()


print(f'=========================Start Generating=========================')

I0 = cv2.imread('/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/data/dataset/apple/110_13051_23361_12/frame000001.jpg')
I2 = cv2.imread('/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/data/dataset/apple/110_13051_23361_12/frame000019.jpg') 

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

images = [I0[:, :, ::-1]]
preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(
    i+1)*(1./args.n) for i in range(args.n - 1)], fast_TTA=TTA)
for pred in preds:
    images.append((padder.unpad(pred).detach().cpu().numpy().transpose(
        1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])
images.append(I2[:, :, ::-1])
mimsave('example/out_Nx.gif', images, fps=args.n)


print(f'=========================Done=========================')
