import cv2
import os
import os.path as osp
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from config import *
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision.utils import save_image
cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CO3dDataset(Dataset):
    def __init__(self, root, tg_frames, in_size, multi, train):
        self.train = 'train' if train else 'test'
        self.root = root
        self.in_size = in_size
        self.tg_frames = tg_frames
        self.multi = multi
        self.objects = glob(self.root+f'/{self.train}/*')
        self.packs = self.get_packs(self.objects)
        if train:
            random.shuffle(self.packs)
        # self.triplets = self.get_triplets(self.objects, tg_frames)
        self.h = 256  # ?????????????????
        self.w = 448    

    def get_packs(self, objects):
        packs = []
        for obj in objects:
            ims = natsorted(glob(obj+'/images/*'))
            pack_size = len(ims)//self.tg_frames
            for i in range(0, len(ims)-pack_size):
                packs.append(ims[i:i+pack_size])
        return packs

    # def is_black(self, frame):
    #     return frame.mean() < 20

    # def extract_triplets_from_object(self, total_frames, frames):
    #     total_imgs = []
    #     # for path in total_frames:
    #     #     img = cv2.imread(path)
    #     #     if not self.is_black(img):
    #     #         total_imgs.append(img)

    #     triplets = []
    #     jump = len(total_imgs)//frames
    #     for i in range(0, len(total_imgs)-jump):
    #         triplets.append(
    #             [total_imgs[i], total_imgs[i+jump//2], total_imgs[i+jump]])
    #     return triplets

    # def get_triplets(self, objects, frames):
    #     triplets = []

    #     futures = []
    #     with ThreadPoolExecutor() as exe:
    #         for obj in objects:
    #             total_frames = natsorted(glob(osp.join(obj, 'images/*')))
    #             futures.append(exe.submit(self.extract_triplets_from_object,
    #                                       total_frames, frames))
    #         for fut in tqdm(as_completed(futures), desc=f'Loading {self.train} Data Objects', total=len(objects)):
    #             triplets.extend(fut.result())

    #     return triplets

    def aug(self, img0, gt, img1, h, w, train=True):
        img0 = cv2.resize(img0, (w, h))
        gt = cv2.resize(gt, (w, h))
        img1 = cv2.resize(img1, (w, h))
        # ih, iw, _ = img0.shape
        # x = np.random.randint(0, ih - h + 1)
        # y = np.random.randint(0, iw - w + 1)
        # img0 = img0[x:x+h, y:y+w, :]
        # img1 = img1[x:x+h, y:y+w, :]
        # gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, index):
        # img0, gt, img1 = self.triplets[index]
        pack = self.packs[index]
        ind = np.arange(len(pack))
        if self.multi:
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()
            # print(ind)
            timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)

        else:
            ind = [0, len(ind)//2, len(ind)-1]
            # print(ind)
            timestep = 0.5

        img0 = cv2.imread(pack[ind[0]])
        gt = cv2.imread(pack[ind[1]])
        img1 = cv2.imread(pack[ind[2]])

        img0, gt, img1 = self.aug(img0, gt, img1, self.in_size, self.in_size)

        if self.train:
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
                timestep = 1 - timestep

            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0), timestep


if __name__ == "__main__":

    dataset = CO3dDataset('/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/data/clean/dataset', 18, 256, multi=True, train=True)
    print(dataset.__len__())
    for i in range(5):
        item = dataset.__getitem__(i)
        print(item[0].shape, item[1])
# sampler = DistributedSampler(dataset)
# train_data = DataLoader(dataset, batch_size=2,
#                         num_workers=1, pin_memory=True, drop_last=True)
# for i, sample in enumerate(train_data):
#     print(sample.shape)
#     break


# class VimeoDataset(Dataset):
#     def __init__(self, dataset_name, path, batch_size=32, model="RIFE"):
#         self.batch_size = batch_size
#         self.dataset_name = dataset_name
#         self.model = model
#         self.h = 256
#         self.w = 448
#         self.data_root = path
#         self.image_root = os.path.join(self.data_root, 'sequences')
#         train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
#         test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
#         with open(train_fn, 'r') as f:
#             self.trainlist = f.read().splitlines()
#         with open(test_fn, 'r') as f:
#             self.testlist = f.read().splitlines()
#         self.load_data()

#     def __len__(self):
#         return len(self.meta_data)

#     def load_data(self):
#         if self.dataset_name != 'test':
#             self.meta_data = self.trainlist
#         else:
#             self.meta_data = self.testlist

#     def aug(self, img0, gt, img1, h, w):
#         ih, iw, _ = img0.shape
#         x = np.random.randint(0, ih - h + 1)
#         y = np.random.randint(0, iw - w + 1)
#         img0 = img0[x:x+h, y:y+w, :]
#         img1 = img1[x:x+h, y:y+w, :]
#         gt = gt[x:x+h, y:y+w, :]
#         return img0, gt, img1

#     def getimg(self, index):
#         imgpath = os.path.join(self.image_root, self.meta_data[index])
#         imgpaths = [imgpath + '/im1.png', imgpath +
#                     '/im2.png', imgpath + '/im3.png']

#         img0 = cv2.imread(imgpaths[0])
#         gt = cv2.imread(imgpaths[1])
#         img1 = cv2.imread(imgpaths[2])
#         return img0, gt, img1

#     def __getitem__(self, index):
#         img0, gt, img1 = self.getimg(index)

#         if 'train' in self.dataset_name:
#             img0, gt, img1 = self.aug(img0, gt, img1, 256, 256)
#             if random.uniform(0, 1) < 0.5:
#                 img0 = img0[:, :, ::-1]
#                 img1 = img1[:, :, ::-1]
#                 gt = gt[:, :, ::-1]
#             if random.uniform(0, 1) < 0.5:
#                 img1, img0 = img0, img1
#             if random.uniform(0, 1) < 0.5:
#                 img0 = img0[::-1]
#                 img1 = img1[::-1]
#                 gt = gt[::-1]
#             if random.uniform(0, 1) < 0.5:
#                 img0 = img0[:, ::-1]
#                 img1 = img1[:, ::-1]
#                 gt = gt[:, ::-1]

#             p = random.uniform(0, 1)
#             if p < 0.25:
#                 img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
#                 gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
#                 img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
#             elif p < 0.5:
#                 img0 = cv2.rotate(img0, cv2.ROTATE_180)
#                 gt = cv2.rotate(gt, cv2.ROTATE_180)
#                 img1 = cv2.rotate(img1, cv2.ROTATE_180)
#             elif p < 0.75:
#                 img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

#         img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
#         img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
#         gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
#         return torch.cat((img0, img1, gt), 0)
