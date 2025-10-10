import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
import imageio


def read_raw_image(file_path, width, height, depth, dtype):
    """读取RAW文件并转换为3D图像数组"""
    print(f"Reading 3D RAW file {file_path}.")

    # 读取原始文件数据
    with open(file_path, 'rb') as f:
        raw_data = f.read()

    # 将原始数据转换为 numpy 数组
    image = np.frombuffer(raw_data, dtype=dtype)

    # 检查数据大小是否匹配预期的 3D 形状
    expected_size = width * height * depth
    if image.size != expected_size:
        raise ValueError(f"File size does not match expected dimensions: ({depth}, {height}, {width})")

    # 重塑数组为图像的三维维度 (depth, height, width)
    image = image.reshape((depth, height, width))

    return image


def sharpen_image(image):
    """对3D图像进行三维卷积锐化处理"""
    # 定义三维锐化卷积核
    kernel = np.array([[[0, 0, 0],
                        [0, -1, 0],
                        [0, 0, 0]],

                       [[0, -1, 0],
                        [-1, 9, -1],
                        [0, -1, 0]],

                       [[0, 0,0],
                        [0, -1, 0],
                        [0, 0, 0]]])

    # 对整个3D图像应用卷积
    sharpened_image = convolve(image, kernel, mode='nearest')

    return sharpened_image

def denoise_image(image, sigma=1):
    """对3D图像进行高斯去噪"""
    # 使用三维高斯滤波去噪
    denoised_image = gaussian_filter(image, sigma=sigma)
    return denoised_image


def save_raw_image(file_path, image, dtype):
    """将三维图像保存为RAW文件"""
    print(f"Saving 3D RAW file to {file_path}.")

    # 将图像数据写入RAW文件
    with open(file_path, 'wb') as f:
        image.tofile(f)


# 设置图像参数
width = 512
height = 512
depth = 512
dtype = np.dtype('<i2')  # 16-bit signed, little-endian byte order

# 设置文件路径
file_path = r"D:\Data\result\output.raw"
output_path = r"D:\Data\result\a_output_sharpened.raw"

# 读取RAW图像数据
image = read_raw_image(file_path, width, height, depth, dtype)
image = denoise_image(image, sigma=0.5)

# 对图像进行三维锐化处理
# image = sharpen_image(image)
# # 显示其中一层锐化后的图像
# plt.imshow(image[291], cmap='gray')
# plt.title('Sharpened Image (Layer 0)')
# plt.colorbar()
# plt.show()

# 保存锐化后的三维图像为RAW文件
save_raw_image(output_path, image, dtype)

print(f"Sharpened 3D image saved to {output_path}")
