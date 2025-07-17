import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 构造一组模拟圆点（例如圆形分布）
theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
radius = 100
x = 500 + radius * np.cos(theta)
y = 500 + radius * np.sin(theta)
pts1 = np.vstack([x, y]).T.astype(np.float32)

# 2. 构造已知仿射矩阵：旋转 + 平移
angle_deg = 10
theta_rad = np.deg2rad(angle_deg)
cos_a = np.cos(theta_rad)
sin_a = np.sin(theta_rad)
tx, ty = 12.0, -4.0  # 平移
M_true = np.array([
    [cos_a, -sin_a, tx],
    [sin_a,  cos_a, ty]
], dtype=np.float32)

# 3. 应用仿射变换：pts2 = M_true * pts1
ones = np.ones((pts1.shape[0], 1), dtype=np.float32)
pts1_h = np.hstack([pts1, ones])  # 转换为齐次坐标
pts2 = (M_true @ pts1_h.T).T

# 4. 使用 estimateAffine2D 恢复仿射矩阵
M_pred, inliers = cv2.estimateAffine2D(pts1, pts2)

# 5. 打印比较结果
print("✅ Ground Truth M:")
print(M_true)
print("\n📐 Estimated M:")
print(M_pred)

error = np.abs(M_true - M_pred)
print("\n🔍 Difference (abs):")
print(error)

print("\n最大误差: ", np.max(error))

# 可视化对比
plt.figure(figsize=(6, 6))
plt.plot(pts1[:, 0], pts1[:, 1], 'ro-', label='Original pts1')
plt.plot(pts2[:, 0], pts2[:, 1], 'go-', label='Transformed pts2')
pts2_pred = (pts1_h @ M_pred.T)
plt.plot(pts2_pred[:, 0], pts2_pred[:, 1], 'b.--', label='Recovered Transform')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.title("Affine Transform Validation")
plt.show()
