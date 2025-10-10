import shutil
import numpy as np
import os
from tqdm import tqdm
import glob
import re
from skimage.restoration import denoise_nl_means, estimate_sigma
from concurrent.futures import ThreadPoolExecutor, as_completed

# 读取 RAW 文件的函数
def read_raw_image(file_path, width, height, dtype):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        image = np.frombuffer(raw_data, dtype=dtype)
        # print(len(raw_data))
        if len(raw_data) % dtype.itemsize != 0:
            raise ValueError(f"Buffer size {len(raw_data)} is not a multiple of element size {dtype.itemsize}")
        if image.size != width * height:
            raise ValueError(f"File size does not match expected dimensions: ({height}, {width})")
        return image.reshape((height, width))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# 非局部均值滤波函数
def apply_non_local_means(image, patch_size=2, patch_distance=5, h=1.0):
    """
    使用 skimage 的非局部均值滤波
    - image: 输入图像
    - patch_size: 单个补丁的大小
    - patch_distance: 搜索区域的大小
    - h: 滤波参数，越大则滤波越强
    """
    # 将图像归一化到 [0, 1] 范围
    image_normalized = (image - image.min()) / (image.max() - image.min())

    # 估算图像噪声标准差
    sigma_est = np.mean(estimate_sigma(image_normalized))  # 不传入 multichannel

    # 应用非局部均值滤波
    denoised = denoise_nl_means(
        image_normalized,
        h=h * sigma_est,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
    )

    # 恢复到原始范围
    denoised_rescaled = denoised * (image.max() - image.min()) + image.min()
    return denoised_rescaled.astype(image.dtype)

# 提取文件名中的数字部分
def extract_number(file_name):
    match = re.search(r'ct_.*?_(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')

# 设置图像参数
width = height = 1420
dtype = np.dtype('<u2')
src = r'D:\Data\cbct'
case = 'image4'
directory = os.path.join(src, case)
output_directory = os.path.join(src, case + "f")  # 输出文件夹路径
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
# 创建目标文件夹（如果不存在）
os.makedirs(output_directory, exist_ok=True)

# 获取文件列表并排序
file_list = glob.glob(os.path.join(directory, 'ct_*_*.raw'))
file_list = sorted(file_list, key=extract_number)

# 多线程处理图像并保存为 .raw 文件
def process_image(file_path):
    image_data = read_raw_image(file_path, width, height, dtype)
    if image_data is not None:
        # 非局部均值滤波
        filtered_image = apply_non_local_means(image_data, h=1)
        # 保存为 .raw 文件
        output_path = os.path.join(output_directory, os.path.basename(file_path))
        filtered_image.astype(dtype).tofile(output_path)
        return f"Processed {os.path.basename(file_path)}"
    return f"Failed to process {os.path.basename(file_path)}"

# 使用线程池并行处理
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_image, file_path) for file_path in file_list]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
        print(future.result())

print(f"所有处理后的图像已保存到文件夹: {output_directory}")
