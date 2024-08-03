import os
import shutil

dir = 'SPINACH/unaugmented_automatic_segmentation/preprocessing_images'

all_images = os.listdir(dir)

for filename in all_images:
    if filename.split("_")[1] =='healthy':
        shutil.copy(os.path.join(dir, filename), os.path.join('WIP2/healthy_images', filename))
    else:
        shutil.copy(os.path.join(dir, filename), os.path.join('WIP2/diseased_images', filename))