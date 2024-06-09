import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import albumentations as A
from PIL import Image
import numpy as np
import os


augmentations = [
    ("HorizontalFlip", A.HorizontalFlip(p=0.3)),
    ("VerticalFlip", A.VerticalFlip(p=0.3)),
    ("RandomRotate90", A.RandomRotate90(p=0.3)),
    ("Transpose", A.Transpose(p=0.3)),
    ("RandomBrightnessContrast", A.RandomBrightnessContrast(p=0.2, brightness_limit=0.2, contrast_limit=0.2)),
    ("HueSaturationValue", A.HueSaturationValue(p=0.2, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20)),
    ("RGBShift", A.RGBShift(p=0.2, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20)),
    ("RandomGamma", A.RandomGamma(p=0.2, gamma_limit=(80, 120))),
    ("GaussNoise", A.GaussNoise(p=0.2, var_limit=(10.0, 50.0))),
    ("MotionBlur", A.MotionBlur(p=0.2, blur_limit=(3, 7))),
    ("MedianBlur", A.MedianBlur(p=0.1, blur_limit=3)),
    ("Blur", A.Blur(p=0.1, blur_limit=3)),
    ("CLAHE", A.CLAHE(p=0.2, clip_limit=4.0)),
    ("ColorJitter", A.ColorJitter(p=0.2, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)),
    ("RandomRain", A.RandomRain(p=0.2, brightness_coefficient=0.9, drop_width=1, blur_value=1)),
    ("RandomSnow", A.RandomSnow(p=0.2, brightness_coeff=2.5, snow_point_lower=0.1, snow_point_upper=0.3)),
    ("RandomShadow", A.RandomShadow(p=0.2, num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5)),
    ("RandomFog", A.RandomFog(p=0.2, fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.1)),
    ("GridDistortion", A.GridDistortion(p=0.2, distort_limit=0.3)),
    ("ElasticTransform", A.ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)),
    ("OpticalDistortion", A.OpticalDistortion(p=0.2, distort_limit=0.3, shift_limit=0.3)),
    # ("RandomSizedCrop", A.RandomSizedCrop(p=0.2, min_max_height=(100, 200), height=256, width=256)),
    # ("RandomResizedCrop", A.RandomResizedCrop(p=0.2, height=256, width=256, scale=(0.8, 1.0))),
    ("Resize", A.Resize(height=256, width=256)),
    # ("Normalize", A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.029, 0.024, 0.025))),
]






transform = A.Compose([aug for name, aug in augmentations])

def augment_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                input_path = os.path.join(root, file_name)
                image = np.array(Image.open(input_path))

                applied_augmentations = []
                augmented = image
                for name, aug in augmentations:
                    augmented_image = aug(image=augmented)['image']
                    if not np.array_equal(augmented, augmented_image):
                        applied_augmentations.append(name)
                    augmented = augmented_image
                
                augmentations_str = "_".join(applied_augmentations) if applied_augmentations else "Original"
                base_name, ext = os.path.splitext(file_name)
                new_file_name = f"{base_name}_{augmentations_str}{ext}"
                
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, new_file_name)
                
                Image.fromarray(augmented.astype(np.uint8)).save(output_path)
                print(f"Processed and saved {output_path}")

def get_leaf_folders(root_folder):
    leaf_folders = []
    for root, dirs, _ in os.walk(root_folder):
        if not dirs:
            leaf_folders.append(root)
    return leaf_folders

root_folder = 'preprocessing'
output_base_folder = 'augmentations'

leaf_folders = get_leaf_folders(root_folder)
print("Folders with no children folders:")
for folder in leaf_folders:
    print(folder)
    relative_path = os.path.relpath(folder, root_folder)
    output_path = os.path.join(output_base_folder, relative_path)
    augment_and_save_images(folder, output_path)

