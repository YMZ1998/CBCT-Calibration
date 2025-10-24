import os

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


if __name__ == "__main__":
    path = r'D:\Data\cbct\DR07051\111'
    path2 = r'D:\Data\cbct\DR07051\111'

    A="A.raw"
    B="B.raw"

    files=os.listdir(path)
    files2=os.listdir(path2)
    print(files)

    file1=files[0]
    file2=files2[1]

    filename1 = os.path.join(path, file1, A)
    filename2 = os.path.join(path2, file2, A)
    width, height = 2130, 2130

    image1 = read_raw_image(filename1, width, height)
    image2 = read_raw_image(filename2, width, height)

    print(image1.max(), image1.min())
    print(image2.max(), image2.min())

    diff_image = image1- image2
    print(diff_image.max(), diff_image.min())
    # diff_image[diff_image>3000] = 10000
    # diff_image[diff_image<1000] = 0

    visualize_comparison(image1, image2, diff_image,
                         title1=file1, title2=file2,
                         title3="Difference (Abs)")

