import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
import sys

from Trainer import Model
from dataset import CO3dDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from config import *
import pdb
import datetime

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'



def get_learning_rate(step, args):
    total_steps = 300 * args.step_per_epoch
    warmup_steps = args.step_per_epoch * 5
    peak_lr = 1e-4
    base_lr = 2e-5

    if step < warmup_steps:

        mul = step / warmup_steps
        return base_lr + (peak_lr - base_lr) * mul
    else:

        mul = (np.cos((step - warmup_steps) / (total_steps - warmup_steps) * math.pi) + 1) / 2
        return (peak_lr - base_lr) * mul + base_lr



def train(model, local_rank, batch_size, data_path):
    if local_rank == 0:
        cur_time =  time.time()
        writer = SummaryWriter('log/train_EMAVFI_200_epoch_freezed_{cur_time}')
    step = 0
    nr_eval = 0
    best = 0
    dataset = CO3dDataset(root=data_path, tg_frames=args.tg_frames,
                          in_size=args.train_im_size, multi=args.multi_interpolate, train=True)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8,
                            pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = CO3dDataset(root=data_path, tg_frames=args.tg_frames,
                              in_size=args.train_im_size, multi=args.multi_interpolate, train=False)
    val_data = DataLoader(dataset_val, batch_size=batch_size,
                          pin_memory=True, num_workers=8)
    print('training...')
    # pdb.set_trace()
    # evaluate(model, val_data, nr_eval, local_rank)
    # sys.exit()
    time_stamp = time.time()
    for epoch in range(200):
        sampler.set_epoch(epoch)
        for i, cat_imgs in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            imgs, timestep = cat_imgs
            imgs = imgs.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step,args)
            _, loss_l1, loss_perc, loss_total = model.update(imgs, gt, learning_rate, training=True,timestep = timestep)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss_l1', loss_l1, step)
                writer.add_scalar('loss_perc', loss_perc, step)
                writer.add_scalar('loss_total', loss_total, step)
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e} loss_perc:{:.4e} loss_total:{:.4e}'.format(epoch, i,
                      args.step_per_epoch, data_time_interval, train_time_interval, loss_l1, loss_perc, loss_total))
            step += 1
        nr_eval += 1
        if nr_eval % 10 == 0:
            evaluate(model, val_data, nr_eval, local_rank)
            model.save_model(epoch, local_rank)
        dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank):
    if local_rank == 0:
        writer_val = SummaryWriter('log/validate_EMAVFI_fixed_lr_freezed')

    psnr = []
    for _, cat_imgs in enumerate(val_data):
        imgs, timestep = cat_imgs
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False,timestep=timestep)
        for j in range(gt.shape[0]):
            val =(gt[j] - pred[j])**2
            psnr.append(-10 * math.log10(val.mean().cpu().item()))


    psnr = np.array(psnr).mean()
    if local_rank == 0:
        print(str(nr_eval), psnr)
        writer_val.add_scalar('psnr', psnr, nr_eval)


if __name__ == "__main__":
    # torchrun train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')

    parser.add_argument('--multi_interpolate', default=True, type=bool,
                        help='True if multi interpolation dataloder else single')
    parser.add_argument('--batch_size', default=18,
                        type=int, help='batch size')
    parser.add_argument('--data_path', default='/home/arpitsah/Desktop/vlr_project/dataset/dataset',
                        type=str, help='data path of co3d')
    parser.add_argument('--tg_frames', default=18, type=int,
                        help='number of frames to generate 3D from')
    parser.add_argument('--train_im_size', default=384,
                        type=int, help='training resolution')
    parser.add_argument('--perceptual_loss', default=False,
                        type=bool, help='use perceptual loss if true')
    parser.add_argument('--perceptual_loss_with_style', default=False,
                        type=bool, help='use perceptual loss if true')
    parser.add_argument('--freeze',default = True,type = bool,help = "freeze everthing except backbone and unet")

    args = parser.parse_args()
    torch.distributed.init_process_group(
        backend="nccl", world_size=args.world_size)
    # torch.distributed.init_process_group(
    #         backend='nccl',
    #         init_method='env://',
    #         timeout=datetime.timedelta(0, 1800),
    #         world_size=args.world_size,
    #         rank=0,
    #         store=None,
    #         group_name='')
    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = Model(local_rank=args.local_rank, use_perceptual_loss=args.perceptual_loss,use_style=args.perceptual_loss_with_style,freeze = args.freeze)
    train(model, args.local_rank, args.batch_size, args.data_path)