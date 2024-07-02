import os
import shutil
image_dir = 'WIP/sampled_images'
mask_dir = 'WIP/masks'
unmasked_dir = 'WIP/unmasked_images'

images = os.listdir(image_dir)
masks = os.listdir(mask_dir)

for filename in images:
    if filename[:-4]+".json" not in masks:
        shutil.copy(os.path.join(image_dir,filename), unmasked_dir)

        os.remove(os.path.join(image_dir, filename))



