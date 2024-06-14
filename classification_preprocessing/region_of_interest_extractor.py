import os
import cv2
import numpy as np

# Directories
input_dir = 'automatic_segmentation/all_images'
mask_dir = 'automatic_segmentation/v1_masks'
roi_dir = 'classification_preprocessing/roi'
os.makedirs(roi_dir, exist_ok=True)

def extract_roi(image, mask):
    """Extract the region of interest from the image using the mask and save as PNG with transparency."""
    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding box of the largest contour
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the ROI from the original image and mask
        roi_image = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]

        # Create an alpha channel based on the mask
        alpha_channel = (roi_mask * 255).astype(np.uint8)

        # Merge the ROI image with the alpha channel
        roi_png = cv2.merge((roi_image, alpha_channel))
        return roi_png
    else:
        return None

# Process each image and corresponding mask
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(mask_dir, f'mask_{filename}')
        
        if os.path.exists(mask_path):
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is not None and mask is not None:
                roi_png = extract_roi(image, mask)
                
                if roi_png is not None:
                    roi_path = os.path.join(roi_dir, os.path.splitext(filename)[0] + '.png')
                    cv2.imwrite(roi_path, roi_png)

print(f"Regions of interest saved in {roi_dir}")
