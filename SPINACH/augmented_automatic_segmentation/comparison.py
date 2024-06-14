import os
import cv2
import matplotlib.pyplot as plt

input_dir = 'SPINACH/augmented_automatic_segmentation/all_images'
mask_dir_1 = 'SPINACH/augmented_automatic_segmentation/v1_masks'
mask_dir_2 = 'SPINACH/augmented_automatic_segmentation/v2_masks'
mask_dir_3 = 'SPINACH/augmented_automatic_segmentation/v3_masks'

output_dir = 'SPINACH/augmented_automatic_segmentation/comparison_images'
os.makedirs(output_dir, exist_ok=True)

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_mask(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def compare_images(image, mask1, mask2, mask3, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask1, cmap='gray')
    axes[1].set_title('Mask 1')
    axes[1].axis('off')

    axes[2].imshow(mask2, cmap='gray')
    axes[2].set_title('Mask 2')
    axes[2].axis('off')

    axes[3].imshow(mask3, cmap='gray')
    axes[3].set_title('Mask 3')
    axes[3].axis('off')

    plt.savefig(output_path)
    plt.close(fig)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        mask1_path = os.path.join(mask_dir_1, f'mask_{filename}')
        mask2_path = os.path.join(mask_dir_2, f'mask_{filename}')
        mask3_path = os.path.join(mask_dir_3, f'mask_{filename}')
        
        if os.path.exists(mask1_path) and os.path.exists(mask2_path) and os.path.exists(mask3_path):
            image = load_image(image_path)
            mask1 = load_mask(mask1_path)
            mask2 = load_mask(mask2_path)
            mask3 = load_mask(mask3_path)

            output_path = os.path.join(output_dir, f'comparison_{filename}')
            compare_images(image, mask1, mask2, mask3, output_path)

print(f"Comparison images saved in {output_dir}")
