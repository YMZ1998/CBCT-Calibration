import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# 设置图像的宽度、高度和深度（通道数）
width = 512  # 图像宽度
height = 512  # 图像高度
depth = 512  # 深度（例如，图像切片的数量）

def read_raw_3d_image(file_path, width, height, depth):
    # 读取 RAW 三维图像
    with open(file_path, 'rb') as f:
        raw_data = f.read()

    # 将二进制数据转换为 NumPy 数组
    image = np.frombuffer(raw_data, dtype=np.dtype('<i2'))
    image = image.reshape((depth, height, width))  # 根据需要调整顺序

    return image


data_path = r'D:\Data\result'

# 读取两幅三维 RAW 图像
image1 = read_raw_3d_image(os.path.join(data_path, 'output_0.500000.raw'), width, height, depth)
image2 = read_raw_3d_image(os.path.join(data_path, 'result.raw'), width, height, depth)

# 计算三维 SSIM 和 PSNR
# 这里对整个三维体积进行计算，可能需要使用其他方法来处理
# 将三维图像的切片合并为一个二维数组，通常在深度方向上堆叠
# 这可以使用 np.stack 将三维数组转换为一个新的形状
combined_image1 = image1.reshape((depth, height * width))
combined_image2 = image2.reshape((depth, height * width))

# 计算 SSIM 和 PSNR
ssim_value = ssim(combined_image1, combined_image2, data_range=combined_image1.max() - combined_image1.min())
psnr_value = psnr(combined_image1, combined_image2)

# 打印结果
print("Overall SSIM Value:", ssim_value)
print("Overall PSNR Value:", psnr_value)

# 可视化其中一层
slice_index = depth // 2  # 中间切片索引
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Image 1 Slice')
plt.imshow(image1[slice_index], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image 2 Slice')
plt.imshow(image2[slice_index], cmap='gray')
plt.axis('off')

plt.show()
