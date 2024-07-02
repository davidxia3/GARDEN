import cv2
import numpy as np
import os

def create_png_cutout(image_path, mask_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read the mask at {mask_path}")
        return
    
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    cutout = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    cutout[np.where(binary_mask == 0)] = [0, 0, 0, 255]  
    
    if not output_path.lower().endswith('.png'):
        print(f"Error: Output path must end with '.png'. Provided path: {output_path}")
        return
    
    success = cv2.imwrite(output_path, cutout)
    if not success:
        print(f"Error: Could not write the image to {output_path}")
    else:
        print(f"PNG cutout saved to {output_path}")


version = "v6"

image_dir = "SPINACH/unaugmented_automatic_segmentation/preprocessing_images"
mask_dir = "SPINACH/unaugmented_automatic_segmentation/masks/" + version + "u_masks"

images = os.listdir(image_dir)
masks = os.listdir(mask_dir)
images = sorted(images)
masks = sorted(masks)


for image_filename in images:
    image_path = os.path.join("SPINACH/unaugmented_automatic_segmentation/preprocessing_images", image_filename)
    mask_path = os.path.join("SPINACH/unaugmented_automatic_segmentation/masks/"+ version+"u_masks", "mask_" + image_filename)
    output_path = os.path.join("SPINACH/unaugmented_automatic_segmentation/cutouts/"+version+"u_cutouts", "cutout_" + image_filename[0:-4] + ".png")
    create_png_cutout(image_path, mask_path, output_path)
