import os
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

from SymmetryEstimation.utils import read_raw_image


def visualize_images(original, denoised):
    """
    可视化原始图像、去噪后的图像和它们的差异。

    Parameters:
    original (ndarray): 原始图像。
    denoised (ndarray): 去噪后的图像。
    diff (ndarray): 原始图像和去噪后图像的差异。
    """
    diff = original - denoised
    print(diff.min(), diff.max())

    plt.figure(figsize=(18, 6))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    # 去噪后的图像
    plt.subplot(1, 3, 2)
    plt.title("Filtered")
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')

    # 原始图像和去噪后图像的差异
    plt.subplot(1, 3, 3)
    plt.title("Difference (Original - Filtered)")
    plt.imshow(diff, cmap='gray')
    plt.colorbar()
    plt.axis('off')

    # 展示图像
    plt.show()


def visualize_bilateral(raw_file, width, height, sigma_color=25.0, sigma_space=25.0, dtype=np.uint16):
    # 1. 读取 raw
    img = read_raw_image(raw_file, width, height, dtype=dtype)

    # 记录开始时间
    start_time = time.time()

    # 使用双边滤波进行去噪
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    denoised = denoised.astype(dtype)
    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    print(f"Bilateral Filtering Time: {end_time - start_time:.4f} seconds")
    visualize_images(img, denoised)


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0709"
    image_size = 1420
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".raw")][::5]
    sigma = 10
    # 处理第一张 raw 文件并可视化
    visualize_bilateral(os.path.join(data_dir, filenames[0]), image_size, image_size, sigma_color=sigma,
                        sigma_space=sigma, dtype=np.uint16)
