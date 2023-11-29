import sys
sys.path.append('emavfi')
from emavfi.utils import InputPadder
from emavfi.Trainer import Model
import emavfi.config as cfg
import cv2
import os
import torch
import numpy as np
import argparse
from glob import glob
from natsort import natsorted

#https://drive.google.com/file/d/1xam35ckjkepfIK0o8BwqX5-NgwPoR-Dx/view?usp=drive_link
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=12, type=int)
    parser.add_argument('--input_dir', default='/raid/xinyu/vlr/dataset/train/110_13051_23361', type=str)
    parser.add_argument('--output_dir', default='/home/xinyu/16824/project/frame-interpolation-VLR/emavfi/interpolations/110_13051_23361', type=str)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--device', default="cuda:2", type=str)
    args = parser.parse_args()
    return args


def run_interpolation_pair(args, model, I0, I2):
    TTA = True
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
    return images


def run_interpolation(args):
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 4, 4]
    )
    model = Model(-1,use_perceptual_loss=True,device=args.device)
    model.load_model()
    model.eval()
    model.device()

    img_list = glob(args.input_dir+'/*')
    img_path_list = natsorted(img_list)
    img_list = [cv2.imread(im) for im in img_path_list]
    imgs = []
    for i in range(len(img_list)-1):
        if args.verbose:
            print(
                f"Running interpolation on {img_path_list[i]} and {img_path_list[i+1]}")
        if i == 0:
            imgs.extend(run_interpolation_pair(
                args, model, img_list[i], img_list[i+1]))
        # if i == len(img_list)-2:
        #     imgs.extend(run_interpolation_pair(args, model, img_list[i], img_list[i+1])[1:])
        else:
            imgs.extend(run_interpolation_pair(
                args, model, img_list[i], img_list[i+1])[1:])

    os.makedirs(args.output_dir, exist_ok=True)
    for i, im in enumerate(imgs):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{args.output_dir}/{i}.png", im)


def run_3d(args):
    pass


if __name__ == "__main__":
    args = get_args()
    run_interpolation(args)
    run_3d(args)
