import numpy as np
import matplotlib.pyplot as plt


# 场景中的球体定义
def sphere_intersection(ray_origin, ray_direction):
    sphere_center = np.array([0, 0, -5])
    radius = 1.0

    oc = ray_origin - sphere_center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - radius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return None  # 没有相交
    else:
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return ray_origin + t * ray_direction  # 返回交点


# 计算光的衰减
def compute_attenuation(light_intensity, distance):
    # 使用距离平方衰减模型
    return light_intensity / (distance ** 2)


# 创建图像
image_size = 512
image = np.zeros((image_size, image_size, 3))

# 相机位置
camera_origin = np.array([0, 0, 0])
light_intensity = 5.0  # 光源强度

for i in range(image_size):
    for j in range(image_size):
        ray_direction = np.array([(i - image_size / 2) / image_size,
                                  (j - image_size / 2) / image_size,
                                  -1])
        ray_direction /= np.linalg.norm(ray_direction)  # 归一化光线方向

        intersection = sphere_intersection(camera_origin, ray_direction)
        if intersection is not None:
            # 计算光源到交点的距离
            distance = np.linalg.norm(intersection - camera_origin)

            # 计算衰减
            attenuation = compute_attenuation(light_intensity, distance)

            # 计算颜色（考虑衰减）
            color = np.array([1, 0, 0]) * attenuation  # 红色，衰减后
            image[i, j] = np.clip(color, 0, 1)  # 确保颜色值在 [0, 1] 范围内

# 显示图像
plt.imshow(image)
plt.axis('off')
plt.show()
