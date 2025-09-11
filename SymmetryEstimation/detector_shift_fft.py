import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifftshift, fftfreq
from scipy.signal import correlate2d

from utils import read_projection_file, read_raw_image

# ===============================
# 参数设置
# ===============================
data_dir = r"D:\Data\cbct\CBCT0707"
projection_size = [1420, 1420]
target_angles = [0, 180]  # 目标角度

# ===============================
# 读取投影文件和角度
# ===============================
proj_file_list, angle_list = read_projection_file(data_dir)

# 找最接近目标角度的文件
selected_proj_files = []
for target in target_angles:
    closest_idx = min(range(len(angle_list)), key=lambda i: abs(angle_list[i] - target))
    selected_proj_files.append(proj_file_list[closest_idx])
    print(f"最接近 {target}° 的投影: {proj_file_list[closest_idx]}（角度: {angle_list[closest_idx]}°）")

# ===============================
# 读取并预处理图像
# ===============================
image_0 = read_raw_image(selected_proj_files[0], *projection_size)
image_180 = read_raw_image(selected_proj_files[1], *projection_size)

# 可选：反转图像（根据具体投影是否需要）
# image_0 = invert_image(image_0)
# image_180 = invert_image(image_180)

# 归一化到 [0,1]
img0 = cv2.normalize(image_0.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)
img180 = cv2.normalize(image_180.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX)

# 水平翻转 180° 投影
img180_flip = cv2.flip(img180, 1)  # 水平翻转

# ===============================
# 方法A：空间域互相关
# ===============================
corr = correlate2d(img0, img180_flip, mode='same')
max_idx = np.unravel_index(np.argmax(corr), corr.shape)
center = (np.array(corr.shape) - 1) / 2
delta_vu = np.array(max_idx) - center  # [Δv, Δu]
print("空间互相关估计偏移 (u,v) ≈", delta_vu[::-1])


# ===============================
# 方法B：傅里叶相位法
# ===============================
def phase_shift(imgA, imgB):
    F0 = fft2(ifftshift(imgA))
    F1 = fft2(ifftshift(imgB))
    cross_phase = np.angle(F0 * np.conj(F1))

    ku, kv = np.meshgrid(
        fftfreq(imgA.shape[1]),
        fftfreq(imgA.shape[0])
    )
    # 线性拟合 cross_phase ≈ -2π (Δu*ku + Δv*kv)
    A = np.column_stack((ku.ravel(), kv.ravel()))
    b = -cross_phase.ravel() / (2 * np.pi)
    delta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return delta[0], delta[1]  # Δu, Δv


du, dv = phase_shift(img0, img180_flip)
print("傅里叶相位估计偏移 (u,v) ≈", (du, dv))

# ===============================
# 可视化投影对比
# ===============================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("0° 投影")
plt.imshow(img0, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("180° 投影(翻转)")
plt.imshow(img180_flip, cmap='gray')
plt.show()
