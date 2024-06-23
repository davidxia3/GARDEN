import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from numpy.random import randn, randint

def generate_latent_points(latent_dim, n_samples, n_classes=2):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


model = load_model('/content/drive/My Drive/GARDEN/generator_model.h5')
latent_points, labels = generate_latent_points(100, 100)
labels = np.random.randint(0, 2, 100)
X = model.predict([latent_points, labels])
X = (X + 1) / 2.0
X = (X * 255).astype(np.uint8)


def show_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    plt.show()


show_plot(X, 2)