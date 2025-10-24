import numpy as np
import os
import imageio
from tqdm import tqdm
from PIL import Image


def read_raw_image(filename, width, height, dtype=np.uint16):
    file_size = width * height * np.dtype(dtype).itemsize - np.dtype(dtype).itemsize

    with open(filename, 'rb') as f:
        raw_data = f.read(file_size)

    image = np.frombuffer(raw_data, dtype=dtype)
    image = np.append(image, 0)
    image = image.reshape((height, width))

    return image


def normalize_image(image):
    image = image.astype(np.float32)
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) / (image_max - image_min)
    return (normalized_image * 255).astype(np.uint8)


def resize_image(image, target_size=(512, 512)):
    pil_image = Image.fromarray(image)
    pil_image_resized = pil_image.resize(target_size, Image.BILINEAR)
    return np.array(pil_image_resized)


width = 1420
height = 1420
dtype = np.uint16

src = r'C:\Users\DATU\Documents\WeChat Files\wxid_hag56n8m9ejr22\FileStorage\File\2025-03\20250327模体数据'
case = '202503271510'
directory = os.path.join(src, case)
num_images = 360

import glob
import re

file_list = glob.glob(os.path.join(directory, '*.raw'))


def extract_numbers(file_name):
    match = re.search(r'ct\.A\.(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')


paths = sorted(file_list, key=extract_numbers)
print(paths)

images = []
for i in tqdm(range(0, num_images, 5)):
    file_path = os.path.join(directory, paths[i])
    image_data = read_raw_image(file_path, width, height, dtype)
    normalized_image = normalize_image(image_data)
    normalized_image = resize_image(normalized_image, target_size=(512, 512))
    images.append(normalized_image)

gif_path = os.path.join(src, case + '.gif')
imageio.mimsave(gif_path, images, duration=0.1)

print(f"GIF 动画已保存到 {gif_path}")

