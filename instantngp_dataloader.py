import os
import shutil
from glob import glob
from tqdm import tqdm
import numpy as np

path = '/home/iam-loki/Documents/XinyuWang/16824/project/emavfi_data/dataset'  # Path to dataset

# # move subfolders from each class so that 44classes -> 99object folder
# for class_folder in tqdm(os.listdir(path)):
#     class_path = os.path.join(path, class_folder)


#     if os.path.isdir(class_path):

#         for version_folder in os.listdir(class_path):
#             version_path = os.path.join(class_path, version_folder)

#             if os.path.isdir(version_path):
            
#                 new_version_path = os.path.join(path, version_folder)
#                 shutil.move(version_path, new_version_path)

    
#         if not os.listdir(class_path):
#             os.rmdir(class_path)



# /home/iam-loki/Documents/XinyuWang/16824/project/emavfi_data/
# ├── dataset_45/
# │   ├── train/
# │   │   ├── 107_12753_23606/
# │   │   │   └── images/
# │   │   ├── 119_13962_28926/
# │   │   │   └── images/
# │   │   └── [other train subfolders with 45 images each]
# │   └── test/
# │       ├── 245_26182_52130/
# │       │   └── images/
# │       ├── 247_26469_51778/
# │       │   └── images/
# │       └── [other test subfolders with 45 images each]
# ├── dataset_36/
# │   ├── train/
# │   │   ├── 107_12753_23606/
# │   │   │   └── images/
# │   │   ├── 119_13962_28926/
# │   │   │   └── images/
# │   │   └── [other train subfolders with 36 images each]
# │   └── test/
# │       ├── 245_26182_52130/
# │       │   └── images/
# │       ├── 247_26469_51778/
# │       │   └── images/
# │       └── [other test subfolders with 36 images each]
# └── dataset_18/
#     ├── train/
#     │   ├── 107_12753_23606/
#     │   │   └── images/
#     │   ├── 119_13962_28926/
#     │   │   └── images/
#     │   └── [other train subfolders with 18 images each]
#     └── test/
#         ├── 245_26182_52130/
#         │   └── images/
#         ├── 247_26469_51778/
#         │   └── images/
#         └── [other test subfolders with 18 images each]      
def pick_and_save_images(image_paths, num_images, save_folder):
    selected_indices = np.linspace(0, len(image_paths) - 1, num_images, dtype=int)
    selected_images = [image_paths[i] for i in selected_indices]

    images_save_folder = os.path.join(save_folder, 'images')
    if not os.path.exists(images_save_folder):
        os.makedirs(images_save_folder)

    for img_path in selected_images:
        shutil.copy(img_path, images_save_folder)

# Iterate over different numbers of images
for num_images in [45, 36, 18]:
    new_path = f'/home/iam-loki/Documents/XinyuWang/16824/project/emavfi_data/dataset_{num_images}'

    # Iterate over train and test folders
    for split_folder in ['train', 'test']:
        split_path = os.path.join(path, split_folder)
        new_split_path = os.path.join(new_path, split_folder)

        if not os.path.exists(new_split_path):
            os.makedirs(new_split_path)

        # Iterate over each subfolder within train/test
        for sub in tqdm(sorted(os.listdir(split_path))):
            sub_path = os.path.join(split_path, sub)
            ims = sorted(glob(os.path.join(sub_path, 'images', '*.jpg')))

            new_sub_path = os.path.join(new_split_path, sub)
            pick_and_save_images(ims, num_images, new_sub_path)
