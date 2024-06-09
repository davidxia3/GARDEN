import os
import shutil
import random

def copy_random_images(source_folders, destination_folder, num_images_to_sample):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for source_folder in source_folders:
        image_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]
                
        sampled_images = random.sample(image_files, min(len(image_files), num_images_to_sample))
        
        for image_name in sampled_images:
            source_path = os.path.join(source_folder, image_name)
            destination_path = os.path.join(destination_folder, image_name)
            shutil.copy(source_path, destination_path)


def get_leaf_folders(root_folder):
    leaf_folders = []
    
    for root, dirs, files in os.walk(root_folder):
        if not dirs:
            leaf_folders.append(root)
    
    return leaf_folders

root_folder = 'augmentations'
augmentation_folders = get_leaf_folders(root_folder)
root_folder = 'preprocessing'
preprocessing_folders = get_leaf_folders(root_folder)
source_folders = augmentation_folders + preprocessing_folders

destination_folder = 'manual_segmentation/images'

num_images_to_sample = 15

copy_random_images(source_folders, destination_folder, num_images_to_sample)
