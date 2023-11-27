import os
import cv2
import shutil
from glob import glob
import numpy as np
from tqdm import tqdm
from natsort import natsorted


def constructBoundary(mask):

    h, w = mask.shape
    top_row = mask[0, :]
    bottom_row = mask[h-1, :]

    left_col = mask[1:h-1, 0]
    right_col = mask[1:h-1, w-1]
    boundary = np.concatenate(
        (top_row, right_col, bottom_row[::-1], left_col[::-1]))

    return boundary


def isOutside(mask):
    mask_np = np.asarray(mask)
    boundary = constructBoundary(mask_np)

    return np.any(boundary != 0)


def isEmptyFrame(mask):
    mean = mask.mean()
    return mean < 10


path = 'dataset'
total_black = []
total_bdry = []
for obj in os.listdir(path):
    path2 = os.path.join(path, obj)
    for sub in tqdm(os.listdir(path2)):
        path3 = os.path.join(path2, sub)
        ims = glob(path3+'/images/*')
        # masks = glob(path3+'/masks/*')

        bdy = []
        blacks = []
        ims = natsorted(ims)
        i = 0
        for im in ims:
            im_name = im.split('/')[-1].split('.')[0]
            mask_path = os.path.join(path3, 'masks', im_name+'.png')
            # print(mask_path,im)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # im = cv2.imread(im)
            # print(im.shape, mask.shape)
            if isEmptyFrame(mask):
                blacks.append(mask_path)
                print(im, mask_path)
                # os.remove(im)
                # os.remove(mask_path)
                # print("empty")
            # elif isOutside(mask):
            #     bdy.append(mask)
                # print("yes")
            else:
                pass
                # i+=1
            # if i==5:
            # exit()
        # print("there are ",i," images outside boundaries")
        # total_black.append(len(blacks))
        # total_bdry.append(len(bdy))
        # if len(blacks)>70:
        #     print(path3)
        #     exit()
        # print(len(bdy))
        # print(len(blacks))
        # print()
        # print()
# a = np.array(total_black)
# b = np.array(total_bdry)

# print(a.mean(), b.mean())
# print(len(a))
# print()
# print((a > 50).sum())
# print((a > 90).sum())

# print()
# print("---")
# print((b > 10).sum())
# print((b > 20).sum())
# print((b > 30).sum())
# print((b > 50).sum())


# print(i)
