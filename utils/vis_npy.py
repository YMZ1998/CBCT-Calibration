import numpy as np
import pyvista as pv
import os

from matplotlib import pyplot as plt

# === 1. 加载数据 ===
npy_path = "../CBCTCalibration/projections_bilevel.npy"
assert os.path.exists(npy_path), f"{npy_path} 文件不存在！"

volume = np.load(npy_path)  # shape: (N_proj, H, W)

plt.imshow(volume.max(axis=0), interpolation='none')
plt.show()

N_proj = volume.shape[0]
print("原始数据形状: ", volume.shape)

# === 2. 提取珠子点 ===
z_idx, y_idx, x_idx = np.where(volume > 0.1)

# === 3. 设置物理参数 ===
pixel_spacing = 0.1     # mm
R = 50.0               # mm，源到中心距离（或源到探测器一半）
angle_range = 360       # degrees
z_spacing = 1.0         # 可忽略，将 z 映射为角度

# === 4. 将角度索引转换为弧度 ===
theta = 2 * np.pi * z_idx / N_proj  # 每个角度点 -> 弧度 [0, 2π)

# === 5. 将点从直角坐标投影到圆柱面上 ===
x_phys = x_idx * pixel_spacing
y_phys = y_idx * pixel_spacing

# 新的圆柱坐标（r = R，θ = theta）
Xc = R * np.cos(theta) + x_phys * np.cos(theta)
Yc = y_phys
Zc = R * np.sin(theta) + x_phys * np.sin(theta)

points_cylinder = np.vstack((Xc, Yc, Zc)).T  # shape: (N, 3)

# === 6. 创建并显示点云 ===
cloud = pv.PolyData(points_cylinder)

plotter = pv.Plotter()
plotter.add_axes()
plotter.add_text("Cylindrical Projection of Beads", font_size=12)
plotter.set_background("white")

plotter.add_mesh(cloud, color='deepskyblue', point_size=4, render_points_as_spheres=True)
plotter.show()
