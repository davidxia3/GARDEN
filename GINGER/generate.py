import torch
from torchvision.utils import save_image
import os
from model import Generator  # Import your Generator class from the model file

# Define constants
Z_DIM = 100
CHANNELS_IMG = 3
FEATURES_GEN = 16
NUM_CLASSES = 2  # Number of classes you have
IMG_SIZE = 256
GEN_EMBEDDING = 100

# Ensure you have the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the generator model
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
checkpoint = torch.load('GINGER/generator2.pth', map_location=device)
generator.load_state_dict(checkpoint['model_state_dict'])
generator.eval()

# Function to generate images for a specific class
def generate_images(generator, num_images, latent_dim, class_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    labels = torch.tensor([class_label] * num_images, device=device)  # Replace with the actual class label
    with torch.no_grad():
        generated_images = generator(noise, labels)
    for i in range(num_images):
        save_image(generated_images[i], os.path.join(output_dir, f'image_{i}_class_{class_label}.png'))

# Parameters
num_images = 100          # Number of images to generate
latent_dim = 100         # Size of the latent vector (depends on your model)
output_dir = 'GINGER/generated_images'  # Directory to save generated images
class_label = 0          # Specify the class label you want to generate

# Generate images for a specific class
generate_images(generator, num_images, latent_dim, class_label, output_dir)

print(f"Generated {num_images} images of class {class_label} and saved to {output_dir}")
