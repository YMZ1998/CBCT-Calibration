import numpy as np
import matplotlib.pyplot as plt
import os

from SymmetryEstimation.utils import read_raw_image

# 设置图像参数
width = 512  # 图像宽度
height = 512  # 图像高度
dtype = np.dtype('<i2')  # 16-bit signed, little-endian byte order ('<i2' 是 numpy 的表示方式)

# 设置 RAW 文件目录和文件名格式
directory = r'D:\Data\result'
num_images = 180  # 假设有100张图像，文件名为 output_1.raw, output_2.raw, ...
file_prefix = 'output_'

# 读取并获取图像数据（间隔 5 张读取一张）
images = []
for i in range(10, num_images, 5):  # 步长为 5
    file_path = os.path.join(directory, f"{file_prefix}{i}.raw")
    image_data = read_raw_image(file_path, width, height, dtype)
    images.append(image_data)

# 显示读取的图像（5xN排列）
num_samples = len(images)  # 实际读取的图像数量
cols = 5  # 每行显示的图像数量
rows = (num_samples + cols - 1) // cols  # 计算所需的行数

fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 3))
axs = axs.flatten()  # 将二维数组展平

# 显示所有读取的图像
for ax in axs:
    ax.axis('off')  # 关闭坐标轴

for ax, image in zip(axs[:num_samples], images):  # 只显示读取的图像
    ax.imshow(image, cmap='gray')

plt.tight_layout()  # 自动调整子图参数
plt.show()
