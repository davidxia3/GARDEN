import torch
from torchvision.utils import save_image
import os
from model import Generator  

Z_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 16
NUM_CLASSES = 2  
IMG_SIZE = 256
GEN_EMBEDDING = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
checkpoint = torch.load('GINGER/generators/generator4_epoch-500.pth', map_location=device)
generator.load_state_dict(checkpoint['model_state_dict'])
generator.eval()

def generate_images(generator, num_images, latent_dim, class_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    labels = torch.tensor([class_label] * num_images, device=device) 
    with torch.no_grad():
        generated_images = generator(noise, labels)
    for i in range(num_images):
        save_image(generated_images[i], os.path.join(output_dir, f'image_{i}_class_{class_label}.png'))

num_images = 100       
latent_dim = 100         
output_dir = 'GINGER/generated_images' 
class_label = 0       

generate_images(generator, num_images, latent_dim, class_label, output_dir)

print(f"Generated {num_images} images of class {class_label} and saved to {output_dir}")
