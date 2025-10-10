import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_raw_image(filename, width, height, dtype=np.uint16):
    file_size = width * height * np.dtype(dtype).itemsize

    with open(filename, 'rb') as f:
        raw_data = f.read(file_size)

    image = np.frombuffer(raw_data, dtype=dtype)
    image = image.reshape((height, width)).astype(np.float32)

    # image = np.clip(image, 0, image.max()*0.45)

    return image


def visualize_comparison(original1, original2, diff, title1="Original1", title2="Original2", title3="Diff"):
    plt.figure(figsize=(15, 5))

    # 显示原始图像1
    plt.subplot(1, 3, 1)
    plt.imshow(original1, cmap='gray')
    plt.title(title1)
    plt.colorbar()
    plt.axis('off')

    # 显示原始图像2
    plt.subplot(1, 3, 2)
    plt.imshow(original2, cmap='gray')
    plt.title(title2)
    plt.colorbar()
    plt.axis('off')

    # 显示图像差异
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='gray')
    plt.title(title3)
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def normalize(image):
    minval, maxval = image.min(), image.max()
    print(f"原始图像 min/max: {minval}, {maxval}")

    normalized_image = (image - minval) / (maxval - minval)
    return normalized_image

if __name__ == "__main__":

    filename1 = r'D:\Data\cbct\DR0401\010\1\A.raw'
    filename2 = r'D:\Data\cbct\CBCT0402\010\202504021632\ct.A.136.46.07.raw'




    width, height = 2130, 2130

    image1 = read_raw_image(filename1, 2130, 2130)
    image2 = read_raw_image(filename2, 1420, 1420)
    image2 = cv2.resize(image2, (2130, 2130), interpolation=cv2.INTER_CUBIC)

    # image1=normalize(image1)
    # image2=normalize(image2)


    print(image1.max(), image1.min())
    print(image2.max(), image2.min())

    diff_image = image1- image2
    print(diff_image.max(), diff_image.min())

    visualize_comparison(image1, image2, diff_image,
                         title1=filename1, title2=filename2,
                         title3="Difference (Abs)")

