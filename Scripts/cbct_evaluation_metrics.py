import SimpleITK as sitk
import numpy as np


def read_mhd_image(file_path):
    """ 读取 MHD 文件并转换为 NumPy 数组 """
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # 获取 3D 数组 [depth, height, width]
    return array.astype(np.float32)


def compute_snr(image):
    """ 计算整张图像的信噪比 SNR """
    image = image - np.min(image)
    mean_val = np.mean(image)  # 计算均值
    std_val = np.std(image)  # 计算标准差

    print("mean_val:", mean_val)
    print("std_val:", std_val)

    if std_val == 0:  # 避免除零错误
        return float('inf')

    return mean_val / std_val  # SNR = 均值 / 标准差


# 读取 CBCT 3D 图像
# image_path = r"D:\Data\cbct\data\result\without_air_norm\A_output_1024.mhd"
image_path = r"D:\Data\cbct\HEAD\ct.mhd"
image = read_mhd_image(image_path)

# 计算 SNR
snr_score = compute_snr(image)
print(f"SNR Score: {snr_score:.4f}")
