import os
import cv2
import matplotlib.pyplot as plt

input_dir = 'SPINACH/unaugmented_automatic_segmentation/preprocessing_images'
mask_dir_1 = 'SPINACH/unaugmented_automatic_segmentation/masks/v1u_masks'
mask_dir_2 = 'SPINACH/unaugmented_automatic_segmentation/masks/v2u_masks'
mask_dir_3 = 'SPINACH/unaugmented_automatic_segmentation/masks/v3u_masks'
mask_dir_4 = 'SPINACH/unaugmented_automatic_segmentation/masks/v4u_masks'
mask_dir_5 = 'SPINACH/unaugmented_automatic_segmentation/masks/v5u_masks'
mask_dir_6 = 'SPINACH/unaugmented_automatic_segmentation/masks/v6u_masks'


output_dir = 'SPINACH/unaugmented_automatic_segmentation/comparison_images'
os.makedirs(output_dir, exist_ok=True)

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_mask(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def compare_images(image, mask1, mask2, mask3, mask4, mask5, mask6, output_path):
    fig, axes = plt.subplots(1, 7, figsize=(35, 5))
    
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

    axes[4].imshow(mask4, cmap='gray') 
    axes[4].set_title('Mask 4')
    axes[4].axis('off')

    axes[5].imshow(mask5, cmap='gray')
    axes[5].set_title('Mask 5')
    axes[5].axis('off')

    axes[6].imshow(mask6, cmap='gray')
    axes[6].set_title('Mask 6')
    axes[6].axis('off')

    plt.savefig(output_path)
    plt.close(fig)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        mask1_path = os.path.join(mask_dir_1, f'mask_{filename}')
        mask2_path = os.path.join(mask_dir_2, f'mask_{filename}')
        mask3_path = os.path.join(mask_dir_3, f'mask_{filename}')
        mask4_path = os.path.join(mask_dir_4, f'mask_{filename}')
        mask5_path = os.path.join(mask_dir_5, f'mask_{filename}')
        mask6_path = os.path.join(mask_dir_6, f'mask_{filename}')
        
        if os.path.exists(mask1_path) and os.path.exists(mask2_path) and os.path.exists(mask3_path) and os.path.exists(mask4_path) and os.path.exists(mask5_path) and os.path.exists(mask6_path):
            image = load_image(image_path)
            mask1 = load_mask(mask1_path)
            mask2 = load_mask(mask2_path)
            mask3 = load_mask(mask3_path)
            mask4 = load_mask(mask4_path)
            mask5 = load_mask(mask5_path)
            mask6 = load_mask(mask6_path)

            output_path = os.path.join(output_dir, f'comparison_{filename}')
            compare_images(image, mask1, mask2, mask3, mask4, mask5, mask6, output_path)

print(f"Comparison images saved in {output_dir}")
