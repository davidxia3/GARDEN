import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_path = 'SPINACH/unaugmented_automatic_segmentation/models/v6u_lr-4_epoch-100.hdf5'

input_dir = 'SPINACH/unaugmented_automatic_segmentation/preprocessing_images'

output_dir = 'SPINACH/unaugmented_automatic_segmentation/masks/v6u_masks'

os.makedirs(output_dir, exist_ok=True)

model = load_model(model_path)

def segment_image(model, image):

    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (256, 256))
    image = image / 255.0 
    image = np.expand_dims(image, axis=-1) 
    image = np.expand_dims(image, axis=0) 

    mask = model.predict(image)

    mask = (mask > 0.5).astype(np.uint8) 
    mask = np.squeeze(mask) 

    return mask

for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)

            if image is not None:
                mask = segment_image(model, image)

                mask_path = os.path.join(output_dir, f'mask_{filename}')
                cv2.imwrite(mask_path, mask * 255) 
            else:
                print(f"Error reading image {image_path}")

print(f"Segmentation masks saved in {output_dir}")
