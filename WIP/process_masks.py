import os
import json
import numpy as np
import cv2

def process_masks(image_dirs, mask_dirs, save_dirs):
    for i, (image_dir, mask_dir, save_dir) in enumerate(zip(image_dirs, mask_dirs, save_dirs)):
        image_files = os.listdir(image_dir)
        mask_files = os.listdir(mask_dir)
        
        for image_file in image_files:
            if image_file[:-4] + '.json' not in mask_files:
                print(f"No mask found for {image_file} in {mask_dir}. Skipping...")
                continue
            
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file[:-4] + '.json')
            
            with open(mask_path, 'r') as f:
                mask_data = json.load(f)["shapes"]
                image = cv2.imread(image_path)
                mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                for shape in mask_data:
                    points = np.array(shape["points"], dtype=np.int32)
                    #label = shape["label"]
                    # if label == "healthy":
                    #     color = (0, 255, 0)
                    # else:
                    #     color = (0, 0, 255) 
                    color = (255,255,255)
                    cv2.fillPoly(mask, [points], color)
                    
                os.makedirs(save_dir, exist_ok=True)
                mask_filename = os.path.splitext(image_file)[0] + '.png'
                cv2.imwrite(os.path.join(save_dir, mask_filename), mask)


image_dirs = ['WIP/sampled_images']
mask_dirs = ['WIP/masks']
save_dirs = ['WIP/processed_masks']
process_masks(image_dirs, mask_dirs, save_dirs)