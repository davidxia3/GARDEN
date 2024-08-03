import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('LETTUCE/unaugmented_models/model_1.h5')

directory = 'SCALLION/isolated_leaves/v2'

images = os.listdir(directory)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  
    return img

c0 = 0
c1 = 0
for img_name in images:
    img_path = os.path.join(directory, img_name)
    
    img = preprocess_image(img_path)
    
    predictions = model.predict(img)
    
    predicted_class = np.argmax(predictions, axis=-1)[0]
    if predicted_class == 0:
        c0 = c0+1
    else:
        c1 = c1+1
    
    print(f"Image: {img_name}, Predicted Class: {predicted_class}")

print("overall")
print(c0)
print(c1)
