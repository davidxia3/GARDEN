import cv2
import numpy as np
import os

image_dir = 'SCALLION/processed_images/'
mask_dir = 'SCALLION/automatic_masks/v2'
images = os.listdir(image_dir)
masks = os.listdir(mask_dir)
for image_filename in images:
    mask_image_filename = 'mask_' + image_filename

    original_image = cv2.imread(os.path.join(image_dir, image_filename))
    mask_image = cv2.imread(os.path.join(mask_dir, mask_image_filename), cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contoured_image = original_image.copy()
    cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)

    # cv2.imshow('Contours', contoured_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    size_threshold = 5000

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        if cv2.contourArea(contour) < size_threshold:
            continue 

        leaf = original_image[y:y+h, x:x+w]
        
        black_background = np.zeros((256, 256, 3), dtype=np.uint8)
        
        contour_mask = np.zeros(mask_image.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        leaf_mask = contour_mask[y:y+h, x:x+w]
        leaf_mask = cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR)  
        
        leaf_height, leaf_width = leaf.shape[:2]
        scale = min(256 / leaf_width, 256 / leaf_height)
        new_size = (int(leaf_width * scale), int(leaf_height * scale))
        resized_leaf = cv2.resize(leaf, new_size, interpolation=cv2.INTER_AREA)
        resized_leaf_mask = cv2.resize(leaf_mask, new_size, interpolation=cv2.INTER_AREA)
        
        start_x = (256 - new_size[0]) // 2
        start_y = (256 - new_size[1]) // 2

        black_background[start_y:start_y+new_size[1], start_x:start_x+new_size[0]] = \
            np.where(resized_leaf_mask == 255, resized_leaf, black_background[start_y:start_y+new_size[1], start_x:start_x+new_size[0]])

        cv2.imwrite(f'SCALLION/isolated_leaves/v2/leaf_{i+1}_'+image_filename, black_background)

    print('Isolated leaves have been saved in the "isolated_leaves" directory.')

