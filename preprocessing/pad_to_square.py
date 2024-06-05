import os
import cv2
import numpy as np

def pad_to_square(image):
    height, width, _ = image.shape
    if height == width:
        return image
    size = max(height, width)
    padded_image = np.full((size, size, 3), 255, dtype=np.uint8)
    x_offset = (size - width) // 2
    y_offset = (size - height) // 2
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
    return padded_image

def resize_to_target(image, target_size):
    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized_image

def process_images(input_dir, output_dir, target_size=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg')):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:
                padded_image = pad_to_square(image)
                resized_image = resize_to_target(padded_image, target_size)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, resized_image)

input_directory = 'raw_data/unique_diseased_images'
output_directory = 'preprocessing/padded_diseased_images'
process_images(input_directory, output_directory, target_size=512)
