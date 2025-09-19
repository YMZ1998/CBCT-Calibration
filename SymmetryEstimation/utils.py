import os

import cv2
import numpy as np


def read_raw_image(filename, width, height, dtype=np.uint16):
    # print(f"正在读取 {filename}...")
    file_size = width * height * np.dtype(dtype).itemsize
    with open(filename, 'rb') as f:
        raw_data = f.read(file_size)
    image = np.frombuffer(raw_data, dtype=dtype)
    image = image.reshape((height, width)).astype(np.float32)
    return image


def parse_angle(filename):
    import re

    match = re.search(r"([-+]?\d+\.\d+)\.raw", filename)
    angle = None
    if match:
        angle = float(match.group(1))

    # print(f"读取 {filename} 完成, 角度: {angle}")

    return angle


def read_projection_file(proj_folder):
    raw_files = sorted([f for f in os.listdir(proj_folder) if f.endswith(".raw")])
    # print(raw_files)
    proj_file = []
    angles = []
    for i, f in enumerate(raw_files):
        angle = parse_angle(os.path.join(proj_folder, f))
        if "A" in f:
            angle = angle + 45
        else:
            angle = angle - 45
        angle = np.fmod(angle + 360.0, 360.0)  # Normalize angle to [0, 360)
        angle = 360.0 - angle  # Flip the angle
        # angle = np.deg2rad(angle)
        proj_file.append(os.path.join(proj_folder, f))
        angles.append(angle)

    return proj_file, angles


def invert_image(image, max_val=1000):
    """
    图像反转（负片），先归一化再反转，输出范围为 [0, max_val]

    参数:
        image (ndarray): 输入图像
        max_val (float): 输出图像的最大值（反转后亮度最大）

    返回:
        ndarray: 反转后的图像
    """
    image = image.astype(np.float32)
    img_min = np.min(image)
    img_max = np.max(image)
    print(f"{img_min}, {img_max}")

    if img_max - img_min < 1e-6:
        norm = np.zeros_like(image)
    else:
        norm = (image - img_min) / (img_max - img_min) * max_val

    inverted = max_val - norm
    return np.clip(inverted, 0, max_val).astype(np.float32)


def compute_metrics(image1, image2, metric='ncc'):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    if metric == 'ncc':
        score = np.mean(image1 * image2)

    elif metric == 'mse':
        score = np.mean((image1 - image2) ** 2)

    elif metric == 'grad_ncc':
        grad1 = cv2.Sobel(image1, cv2.CV_32F, 1, 0, ksize=3)
        grad2 = cv2.Sobel(image2, cv2.CV_32F, 1, 0, ksize=3)
        mean1 = np.mean(grad1)
        mean2 = np.mean(grad2)
        numerator = np.sum((grad1 - mean1) * (grad2 - mean2))
        denominator = np.sqrt(np.sum((grad1 - mean1) ** 2) * np.sum((grad2 - mean2) ** 2))
        score = numerator / (denominator + 1e-8)

    elif metric == 'ssim':
        from skimage.metrics import structural_similarity as ssim
        # 将图像拉回 [0,1] 区间，因为 skimage 的 ssim 要求如此
        img1_norm = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8)
        img2_norm = (image2 - image2.min()) / (image2.max() - image2.min() + 1e-8)
        score = ssim(img1_norm, img2_norm, data_range=1.0)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return score


def normalize_image(image):
    """标准化图像数据"""
    # return (image - np.mean(image)) / (np.std(image) + 1e-5)
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)
