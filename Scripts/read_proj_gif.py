import numpy as np
import os
import imageio
from tqdm import tqdm
from PIL import Image  # 导入 PIL 库进行图像处理


# 读取 RAW 文件的函数
def read_raw_image(file_path, width, height, dtype):
    # 读取原始文件数据
    with open(file_path, 'rb') as f:
        raw_data = f.read()

    # 将原始数据转换为 numpy 数组
    image = np.frombuffer(raw_data, dtype=dtype)
    if image.size != width * height:
        print(file_path)
        raise ValueError(f"File size does not match expected dimensions: ({height}, {width})")
    # 重塑数组为图像的维度 (height, width)
    image = image.reshape((height, width))
    return image


# 归一化图像到 0-255 的函数
def normalize_image(image):
    # 将图像转换为浮点数
    image = image.astype(np.float32)
    # 归一化到 0-1
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) / (image_max - image_min)  # 归一化到 [0, 1]
    # 转换到 0-255 并返回为 uint8 类型
    return (normalized_image * 255).astype(np.uint8)


# 图像缩放函数
def resize_image(image, target_size=(512, 512)):
    # 使用 PIL 进行缩放
    pil_image = Image.fromarray(image)  # 将 numpy 数组转换为 PIL 图像
    pil_image_resized = pil_image.resize(target_size, Image.BILINEAR)  # 使用高质量缩放
    return np.array(pil_image_resized)  # 将 PIL 图像转换回 numpy 数组


# 设置图像参数
width = 2130  # 图像宽度
height = 2130  # 图像高度
dtype = np.dtype('<i2')  # 16-bit signed, little-endian byte order ('<i2' 是 numpy 的表示方式)

# 设置 RAW 文件目录和文件名格式
src = r'D:\Data\cbct'
case = 'image2'
directory = os.path.join(src, case)
num_images = 360

import glob
import re

file_list = glob.glob(os.path.join(directory, 'ct_*_*.raw'))


def extract_number(file_name):
    match = re.search(r'ct_.*?_(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')

paths = sorted(file_list, key=extract_number)
print(paths)

# 读取并获取图像数据（间隔 5 张读取一张）
images = []
for i in tqdm(range(0, num_images, 5)):  # 步长为 5
    file_path = os.path.join(directory, paths[i])
    image_data = read_raw_image(file_path, width, height, dtype)
    normalized_image = normalize_image(image_data)  # 归一化图像
    normalized_image = resize_image(normalized_image, target_size=(512, 512))  # 缩放图像
    images.append(normalized_image)

# 生成 GIF 动画
gif_path = os.path.join(src, case + '.gif')
imageio.mimsave(gif_path, images, duration=0.1)  # duration 设置每帧的持续时间（单位：秒）

print(f"GIF 动画已保存到 {gif_path}")
