import numpy as np
import matplotlib.pyplot as plt

from SymmetryEstimation.utils import read_raw_image


def visualize_image(image1, image2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(image1, cmap='gray')
    axs[0].axis('off')

    axs[1].imshow(image2, cmap='gray')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def apply_gamma_correction(image, gamma=1.0):
    image_normalized = (image.astype(np.float32)-np.min(image))/ (np.max(image)-np.min(image))
    corrected = 1-np.power(image_normalized, gamma)
    # plt.imshow(corrected, cmap='gray')
    # print(corrected.max())
    return (corrected * np.max(image)).astype(np.uint16)

if __name__ == "__main__":
    filename = r"D:\\Data\\cbct\\move_after\\DR1\\A.raw"

    width, height = 4260, 4260

    image = read_raw_image(filename, width, height, dtype=np.uint16)
    print("图像形状:", image.shape)

    image2=image.copy()
    image2=np.clip(image2, 0, 9000)


    result = apply_gamma_correction(image2, gamma=0.5)

    # result = np.max(image2) - image2
    # print(result.shape, result.dtype)
    visualize_image(image, result)
