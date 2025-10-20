import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from SymmetryEstimation.utils import read_raw_image


def display_mean_and_animation(data_dir, image_size):
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".raw")]

    images = []
    for idx, fname in enumerate(sorted(filenames)):
        filename = os.path.join(data_dir, fname)
        image = read_raw_image(filename, image_size, image_size)
        images.append(image)

    mean_image = np.mean(images, axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(mean_image, cmap='gray')  # Display mean image in grayscale
    plt.colorbar()  # Add colorbar to visualize intensity
    plt.title("Mean Image")
    plt.axis('off')  # Hide axes
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Function to update the displayed image in animation
    def update_frame(i):
        ax.clear()  # Clear previous image
        ax.imshow(images[i], cmap='gray')  # Show the current image
        ax.axis('off')  # Hide axes
        ax.set_title(f"Frame {i + 1}")

    ani = animation.FuncAnimation(fig, update_frame, frames=len(images), interval=100, repeat=True)
    plt.show()


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\air_b"  # Path to your raw images
    image_size = 1420  # Ensure this matches your image dimensions

    # Display mean image and animation of raw images
    display_mean_and_animation(data_dir, image_size)
