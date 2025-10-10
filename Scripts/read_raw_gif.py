import numpy as np
import os
import imageio
from tqdm import tqdm

def read_raw_image(file_path, width, height, dtype):
    with open(file_path, 'rb') as f:
        raw_data = f.read()

    image = np.frombuffer(raw_data, dtype=dtype)
    if image.size != width * height:
        print(file_path)
        raise ValueError(f"File size does not match expected dimensions: ({height}, {width})")
    image = image.reshape((height, width))

    return image

def normalize_image(image):
    image = image.astype(np.float32)
    image_min = np.min(image)
    image_max = np.max(image)
    normalized_image = (image - image_min) / (image_max - image_min)
    return (normalized_image * 255).astype(np.uint8)

width = height = 1024
dtype = np.dtype('<i2')


directory = r'D:\Data\result'
num_images = 360
file_prefix = 'output_'

images = []
for i in tqdm(range(0, num_images, 3)):
    file_path = os.path.join(directory, f"{file_prefix}{i}.raw")
    image_data = read_raw_image(file_path, width, height, dtype)
    normalized_image = normalize_image(image_data)
    images.append(normalized_image)

gif_path = os.path.join(directory, f'output_{width}.gif')
imageio.mimsave(gif_path, images, duration=0.1)
print(f"GIF 动画已保存到 {gif_path}")
