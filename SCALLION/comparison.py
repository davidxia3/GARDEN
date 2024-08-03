import os
import cv2
import matplotlib.pyplot as plt

input_dir = 'WIP/processed_images'
mask_dir_1 = 'WIP/automatic_masks/v1'
mask_dir_2 = 'WIP/automatic_masks/v2'
mask_dir_3 = 'WIP/automatic_masks/v3'
mask_dir_5 = 'WIP/automatic_masks/v5'



output_dir = 'WIP/comparison_images'
os.makedirs(output_dir, exist_ok=True)

def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

def load_mask(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def compare_images(image, mask1, mask2, mask3, mask5, output_path):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
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

    axes[4].imshow(mask5, cmap='gray')
    axes[4].set_title('Mask 5')
    axes[4].axis('off')

    plt.savefig(output_path)
    plt.close(fig)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        mask1_path = os.path.join(mask_dir_1, f'mask_{filename}')
        mask2_path = os.path.join(mask_dir_2, f'mask_{filename}')
        mask3_path = os.path.join(mask_dir_3, f'mask_{filename}')
        mask5_path = os.path.join(mask_dir_5, f'mask_{filename}')

        
        if os.path.exists(mask1_path) and os.path.exists(mask2_path) :
            image = load_image(image_path)
            mask1 = load_mask(mask1_path)
            mask2 = load_mask(mask2_path)
            mask3 = load_mask(mask2_path)
            mask5 = load_mask(mask2_path)


            output_path = os.path.join(output_dir, f'comparison_{filename}')
            compare_images(image, mask1, mask2, mask3, mask5,  output_path)

print(f"Comparison images saved in {output_dir}")
