import os
import shutil

import cv2
import numpy as np

from bilateral_raw import visualize_images
from SymmetryEstimation.utils import read_raw_image


def denoise_raw_file(raw_file, width, height, sigma_color=25.0, sigma_space=25.0, dtype=np.uint16):
    """
    读取单张 raw 文件并用双边滤波去噪
    """
    img = read_raw_image(raw_file, width, height, dtype=dtype)

    # 使用双边滤波进行去噪
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    denoised = denoised.astype(dtype)

    # 可视化
    visualize_images(img, denoised)

    # 返回去噪后的图像
    return denoised


def process_folder(input_dir, output_dir, width, height, sigma_color=25.0, sigma_space=25.0, dtype=np.uint16):
    """
    批量处理文件夹内 raw 文件，并保存去噪结果
    """
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    filenames = [f for f in os.listdir(input_dir) if f.endswith(".raw")]
    filenames.sort()  # 按文件名排序处理

    for i, fname in enumerate(filenames):
        input_path = os.path.join(input_dir, fname)
        print(f"[{i + 1}/{len(filenames)}] Processing {fname} ...")

        # 读取并去噪
        denoised = denoise_raw_file(input_path, width, height, sigma_color, sigma_space, dtype)

        output_path = os.path.join(output_dir, fname)
        denoised.tofile(output_path)

        print(f"Saved denoised: {output_path}")


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0709"
    output_dir = r"D:\Data\cbct\CBCT0709_denoised2"
    image_size = 1420

    sigma = 5
    process_folder(data_dir, output_dir, image_size, image_size, sigma_color=sigma, sigma_space=sigma, dtype=np.uint16)
