import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def read_raw_image(filename, width, height, dtype=np.uint16):
    file_size = width * height * np.dtype(dtype).itemsize - np.dtype(dtype).itemsize

    with open(filename, 'rb') as f:
        raw_data = f.read(file_size)

    image = np.frombuffer(raw_data, dtype=dtype)
    image = np.append(image, 0)
    image = image.reshape((height, width))

    return image

def calculate_snr(image1, image2):
    """
    计算信噪比 (SNR)。

    参数:
        image1 (np.ndarray): 第一幅图像（信号）。
        image2 (np.ndarray): 第二幅图像（信号 + 噪声）。

    返回:
        snr (float): 信噪比 (dB)。
    """
    # 计算信号功率
    signal_power = np.mean(image1 ** 2)

    # 计算噪声功率
    noise_image = image1 - image2
    noise_power = np.mean(noise_image ** 2)

    # 计算 SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_psnr(image1, image2, max_pixel=65535):
    """
    计算峰值信噪比 (PSNR)。

    参数:
        image1 (np.ndarray): 第一幅图像。
        image2 (np.ndarray): 第二幅图像。
        max_pixel (int): 图像的最大像素值，默认为 65535（16 位图像）。

    返回:
        psnr (float): 峰值信噪比 (dB)。
    """
    # 计算均方误差 (MSE)
    mse = np.mean((image1 - image2) ** 2)

    # 计算 PSNR
    if mse == 0:
        return float('inf')  # 如果 MSE 为 0，PSNR 为无穷大
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_mse(image1, image2):
    """
    计算均方误差 (MSE)。

    参数:
        image1 (np.ndarray): 第一幅图像。
        image2 (np.ndarray): 第二幅图像。

    返回:
        mse (float): 均方误差。
    """
    mse = np.mean((image1 - image2) ** 2)
    return mse

def calculate_ssim(image1, image2):
    """
    计算结构相似性 (SSIM)。

    参数:
        image1 (np.ndarray): 第一幅图像。
        image2 (np.ndarray): 第二幅图像。

    返回:
        ssim_value (float): 结构相似性。
    """
    ssim_value = ssim(image1, image2, data_range=image1.max() - image1.min())
    return ssim_value

def calculate_noise_metrics(image1, image2):
    """
    计算噪声指标。

    参数:
        image1 (np.ndarray): 第一幅图像。
        image2 (np.ndarray): 第二幅图像。

    返回:
        dict: 包含噪声指标的字典。
    """
    # 计算差分图像
    noise_image = image1.astype(np.float32) - image2.astype(np.float32)

    # 计算噪声水平（标准差）
    noise_level = np.std(noise_image)

    # 计算信噪比 (SNR)
    snr = calculate_snr(image1, image2)

    # 计算峰值信噪比 (PSNR)
    psnr = calculate_psnr(image1, image2)

    # 计算均方误差 (MSE)
    mse = calculate_mse(image1, image2)

    # 计算结构相似性 (SSIM)
    ssim_value = calculate_ssim(image1, image2)

    return {
        "noise_level": noise_level,
        "snr": snr,
        "psnr": psnr,
        "mse": mse,
        "ssim": ssim_value
    }

def visualize_images(image1, image2, noise_image):
    """
    可视化原始图像和噪声图像。

    参数:
        image1 (np.ndarray): 第一幅图像。
        image2 (np.ndarray): 第二幅图像。
        noise_image (np.ndarray): 噪声图像。
    """
    plt.figure(figsize=(15, 5))

    # 显示第一幅图像
    plt.subplot(1, 3, 1)
    plt.imshow(image1, cmap='gray')
    plt.title("Image 1")
    plt.axis('off')

    # 显示第二幅图像
    plt.subplot(1, 3, 2)
    plt.imshow(image2, cmap='gray')
    plt.title("Image 2")
    plt.axis('off')

    # 显示噪声图像
    plt.subplot(1, 3, 3)
    plt.imshow(noise_image, cmap='gray')
    plt.title("Noise Image")
    plt.axis('off')

    plt.show()

# 示例用法
if __name__ == "__main__":
    # 文件路径
    filename1 = r"D:\\Data\\cbct\\Data\\C1.raw"
    filename2 = r"D:\\Data\\cbct\\Data\\C2.raw"

    # 图像尺寸
    width, height = 1420, 1420

    # 读取两幅图像
    image1 = read_raw_image(filename1, width, height, dtype=np.uint16)
    image2 = read_raw_image(filename2, width, height, dtype=np.uint16)

    # 计算噪声指标
    metrics = calculate_noise_metrics(image1, image2)

    # 打印结果
    print("噪声水平 (标准差):", metrics["noise_level"])
    print("信噪比 (SNR):", metrics["snr"])
    print("峰值信噪比 (PSNR):", metrics["psnr"])
    print("均方误差 (MSE):", metrics["mse"])
    print("结构相似性 (SSIM):", metrics["ssim"])

    # 可视化图像
    noise_image = image1.astype(np.float32) - image2.astype(np.float32)
    # print("noise_image", noise_image)
    visualize_images(image1, image2, noise_image)