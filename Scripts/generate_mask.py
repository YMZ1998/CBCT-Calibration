import SimpleITK as sitk
import numpy as np
import os

# 设置参数
input_path = r'D:\Data\MIR\images\mr1.nii.gz'
output_dir = r'D:\Data\MIR\images'
# output_dir = r'C:\Users\DATU\Desktop\validation'
value = 1
dtype = np.uint8

# dtype = np.int16
# value = 255
radius_mm = 30.0
height_mm = radius_mm * 2  # 用于圆柱体

# 读取图像
image = sitk.ReadImage(input_path)
spacing = image.GetSpacing()
size = image.GetSize()
origin = image.GetOrigin()
direction = image.GetDirection()
image_np = sitk.GetArrayFromImage(image)  # shape: (z, y, x)
shape = image_np.shape
center_index = [int(sz / 2) for sz in reversed(size)]  # z, y, x

# 构建坐标网格和物理坐标
zz, yy, xx = np.indices(shape)
coord_x = (xx - center_index[2]) * spacing[0]
coord_y = (yy - center_index[1]) * spacing[1]
coord_z = (zz - center_index[0]) * spacing[2]


def make_cube_mask():
    cube_size_voxel = [int(radius_mm / s) for s in spacing[::-1]]  # z, y, x
    mask = np.zeros(shape, dtype=dtype)
    z1, y1, x1 = center_index
    mask[
    z1 - cube_size_voxel[0]:z1 + cube_size_voxel[0],
    y1 - cube_size_voxel[1]:y1 + cube_size_voxel[1],
    x1 - cube_size_voxel[2]:x1 + cube_size_voxel[2]
    ] = value
    return mask


def make_sphere_mask():
    dist_mm = np.sqrt(coord_x ** 2 + coord_y ** 2 + coord_z ** 2)
    return (dist_mm <= radius_mm).astype(dtype) * value


def make_cylinder_mask():
    dist_xy = np.sqrt(coord_x ** 2 + coord_y ** 2)
    z_min, z_max = -height_mm / 2, height_mm / 2
    mask = ((dist_xy <= radius_mm) &
            (coord_z >= z_min) & (coord_z <= z_max)).astype(dtype) * value
    return mask

def make_star_cylinder_mask(base_radius_mm=radius_mm, star_amp=0.2, num_points=6):
    """
    生成一个具有平滑五角星横截面的柱体掩膜（边缘连续），与图像同大小。

    参数:
        base_radius_mm: 基础半径
        star_amp: 星角扰动振幅（推荐 0.2 ~ 0.5）
        num_points: 星角数量（默认 5）
        height_mm: 柱体的物理高度（单位 mm）
    返回:
        mask: numpy.ndarray，掩膜数组，shape 与 image_np 相同
    """
    # 极坐标角度和平面距离
    angle_xy = np.arctan2(coord_y, coord_x)
    dist_xy = np.sqrt(coord_x ** 2 + coord_y ** 2)

    # 星形半径函数（平滑）
    star_radius = base_radius_mm * (1 + star_amp * np.cos(num_points * angle_xy))

    # 广播到 3D（z 方向复制）
    star_radius_3d = np.broadcast_to(star_radius, image_np.shape)

    # Z 范围限制
    z_min = -height_mm / 2
    z_max = height_mm / 2

    # 掩膜构造
    mask = np.zeros_like(image_np, dtype=dtype)
    mask[(dist_xy <= star_radius_3d) & (coord_z >= z_min) & (coord_z <= z_max)] = value
    return mask


def save_mask(mask_np, filename):
    mask_img = sitk.GetImageFromArray(mask_np)
    mask_img.CopyInformation(image)
    sitk.WriteImage(mask_img, os.path.join(output_dir, filename))


# 生成并保存三个 mask
save_mask(make_cube_mask(), "mask_cube.nii.gz")
save_mask(make_sphere_mask(), "mask_sphere.nii.gz")
save_mask(make_cylinder_mask(), "mask_cylinder.nii.gz")
save_mask(make_star_cylinder_mask(), "make_star_cylinder_mask.nii.gz")
