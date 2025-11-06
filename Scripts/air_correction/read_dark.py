import matplotlib.pyplot as plt

from SymmetryEstimation.utils import read_raw_image


def read_dark_image(filename, image_size):
    image = read_raw_image(filename, image_size, image_size)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')  # Display mean image in grayscale
    plt.colorbar()  # Add colorbar to visualize intensity
    plt.title("Image")
    plt.axis('off')  # Hide axes
    plt.show()


if __name__ == "__main__":
    filename = r"D:\Data\cbct\air_correction\dark\dark_a.raw"
    image_size = 1420

    read_dark_image(filename, image_size)
