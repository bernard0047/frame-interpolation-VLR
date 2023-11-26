import os
import cv2
import shutil
from glob import glob
import numpy as np
from tqdm import tqdm

###test instant ngp on 202frames, 45 frames,36,18


path = '/home/arpitsah/Desktop/Fall-2023/VLR/project/frame-interpolation-VLR/data/dataset'  #path to o3d dataset
total_black = []
total_bdry = []


# Function to pick and save a specified number of images
def pick_and_save_images(image_paths, num_images, save_folder):
    selected_indices = np.linspace(0, len(image_paths) - 1, num_images, dtype=int)
    selected_images = [image_paths[i] for i in selected_indices]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for img_path in selected_images:
        shutil.copy(img_path, save_folder)


for obj in tqdm(os.listdir(path)):
    path2 = os.path.join(path, obj)
    for sub in sorted(os.listdir(path2)):
        path3 = os.path.join(path2, sub)
        ims = sorted(glob(os.path.join(path3, 'images', '*.jpg')))
        # Pick and save images
        for num_images in [9,12,6]:
            save_folder = os.path.join(path2, f"{sub}_{num_images}")
            pick_and_save_images(ims, num_images, save_folder)
        print("subfolder name", obj, sub)
        break