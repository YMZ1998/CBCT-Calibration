import os

import numpy as np


def read_raw_image(filename, width, height, dtype=np.uint16):
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

    print(f"读取 {filename} 完成, 角度: {angle}")

    return angle


def read_projection_file(proj_folder):
    raw_files = sorted([f for f in os.listdir(proj_folder) if f.endswith(".raw")])
    print(raw_files)
    proj_file = []
    angles = []
    for i, f in enumerate(raw_files):
        angle = parse_angle(os.path.join(proj_folder, f))
        angle = angle + 45
        # angles = np.fmod(angles + 360.0, 360.0)  # Normalize angle to [0, 360)
        # angles = 360.0 - angles  # Flip the angle
        # angle = np.deg2rad(angle)
        proj_file.append(os.path.join(proj_folder, f))
        angles.append(angle)

    return proj_file, angles


def invert_image(image, max_val=500):
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

    if img_max - img_min < 1e-6:
        norm = np.zeros_like(image)
    else:
        norm = (image - img_min) / (img_max - img_min) * max_val

    inverted = max_val - norm
    return np.clip(inverted, 0, max_val).astype(np.float32)
