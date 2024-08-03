import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_path = 'SCALLION/models/v5_lr-34_bs-32_epoch-75.hdf5'

input_dir = 'SCALLION/processed_images'

output_dir = 'SCALLION/automatic_masks/v5'

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
