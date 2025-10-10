import numpy as np
import matplotlib.pyplot as plt


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


def save_raw_image(image, file_path):
    """将3D图像数组保存为RAW文件"""
    print(f"Saving 3D RAW file to {file_path}.")
    with open(file_path, 'wb') as f:
        f.write(image.tobytes())


# 设置图像参数
width = height = 1024
depth = 256
# width = height = 1420
# depth = 1
dtype = np.dtype('<i2')

# 读取两个3D RAW文件
file_path1 = r"D:\Data\result\output_1024-1.raw"
file_path2 = r"D:\Data\result\output_1024-2.raw"
# file_path1 = r"D:\Data\result\z_slice_242.raw"
# file_path2 = r"D:\Data\result\z_slice_2422.raw"
# file_path1 = r"D:\Data\cbct\image4\ct_179.87_001.raw"
# file_path2 = r"D:\Data\cbct\image4f\ct_179.87_001.raw"
image1 = read_raw_image(file_path1, width, height, depth, dtype)
image2 = read_raw_image(file_path2, width, height, depth, dtype)

# 计算差异（可以是绝对差或其他方法）
diff_image = np.abs(image1 - image2)
print(np.max(diff_image))

# 将差异图保存为新的RAW文件
output_diff_path = r"D:\Data\result\output_diff.raw"
save_raw_image(diff_image, output_diff_path)
