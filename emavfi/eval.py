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
import config as cfg
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
    total_steps = 100 * args.step_per_epoch  
    warmup_steps = args.step_per_epoch * 5 
    peak_lr = 1e-4  
    base_lr = 2e-5 

    if step < warmup_steps:
        
        mul = step / warmup_steps
        return base_lr + (peak_lr - base_lr) * mul
    else:
        
        mul = (np.cos((step - warmup_steps) / (total_steps - warmup_steps) * math.pi) + 1) / 2
        return (peak_lr - base_lr) * mul + base_lr


def evaluate_model(model, local_rank, batch_size, data_path):
    if local_rank == 0:
        writer = SummaryWriter('log/eval_EMAVFI')
    # step = 0
    nr_eval = 0
    # best = 0
    # dataset = CO3dDataset(root=args.data_path, tg_frames=args.tg_frames,
    #                       in_size=args.train_im_size, multi=args.multi_interpolate, train=True)
    # sampler = DistributedSampler(dataset)
    # train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8,
    #                         pin_memory=True, drop_last=True, sampler=sampler)
    # args.step_per_epoch = train_data.__len__()
    dataset_val = CO3dDataset(root=args.data_path, tg_frames=args.tg_frames,
                              in_size=args.eval_im_size, multi=args.multi_interpolate, train=True)
    val_data = DataLoader(dataset_val, batch_size=batch_size,
                          pin_memory=True, num_workers=8)
    print('Evaluating...')
    # pdb.set_trace()
    # evaluate(model, val_data, nr_eval, local_rank)
    # sys.exit()

    evaluate(model, val_data, nr_eval, local_rank)
    print('Evaluation Complete...')

    
def evaluate(model, val_data, nr_eval, local_rank):

    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=16,
        depth=[2, 2, 2, 2, 2]
    )
    model = Model(-1,use_perceptual_loss= False)
    model.load_model()
    model.eval()
    model.get_device()

    if local_rank == 0:
        writer_val = SummaryWriter('log/validate_EMAVFI_fixed_lr')

    psnr = []
    for _, cat_imgs in enumerate(val_data):
        imgs, timestep = cat_imgs 
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False,timestep=timestep)
        for j in range(gt.shape[0]):
            #  psnr.append(-10 * math.log10(max(1e-10, ((gt[j] - pred[j])**2).mean().cpu().item())))
            val =(gt[j] - pred[j])**2
            psnr.append(-10 * math.log10(val.mean().cpu().item()))


    psnr = np.array(psnr).mean()
    if local_rank == 0:
        print(str(nr_eval), psnr)
        writer_val.add_scalar('psnr', psnr, nr_eval)
def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }

if __name__ == "__main__":
    # torchrun train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')

    parser.add_argument('--multi_interpolate', default=True, type=bool,
                        help='True if multi interpolation dataloder else single')
    parser.add_argument('--batch_size', default=1,
                        type=int, help='batch size')
    parser.add_argument('--data_path', default='/home/arpitsah/Desktop/vlr_project/dataset/dataset',
                        type=str, help='data path of co3d')
    parser.add_argument('--tg_frames', default=18, type=int,
                        help='number of frames to generate 3D from')
    parser.add_argument('--eval_im_size', default=384,
                        type=int, help='training resolution')
    parser.add_argument('--perceptual_loss', default=False,
                        type=bool, help='use perceptual loss if true')
    
    args = parser.parse_args()
    torch.distributed.init_process_group(
        backend="nccl", world_size=args.world_size)

    torch.cuda.set_device(args.local_rank)
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank, args.perceptual_loss)
    evaluate_model(model, args.local_rank, args.batch_size, args.data_path)