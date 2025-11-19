import os

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi

from Scripts.Lattice_RT.create_sphere_mask import create_sphere_mask
from Scripts.Lattice_RT.lattice_distance import create_manual_gtv, compute_lattice_distances


# -------------------------------
# 生成晶格点（Grid排布）
# -------------------------------
def generate_lattice_points_grid(gtv_img, radius=10.0, min_edge_dist_mm=2.0, shrink_mm=0.0):
    """
    使用规则 grid 排布晶格点
    radius: 球半径 (mm)
    min_edge_dist_mm: 球体与 GTV 边缘最小距离
    shrink_mm: 可选择收缩 GTV 的距离
    """
    gtv = sitk.GetArrayFromImage(gtv_img).astype(np.uint8)
    spacing = gtv_img.GetSpacing()  # (sx, sy, sz)
    sz, sy, sx = gtv.shape

    # 可选：收缩 GTV
    if shrink_mm > 0:
        # 使用距离变换收缩
        dt = ndi.distance_transform_edt(gtv, sampling=spacing[::-1])
        mask = dt >= shrink_mm + radius
    else:
        mask = gtv > 0

    # 计算 grid 间距，保证球不重叠
    # 球半径 + 最小间距 => grid step
    step_mm = 2 * radius + min_edge_dist_mm

    # 转换为 voxel 间距
    step_voxel_x = max(1, int(np.round(step_mm / spacing[0])))
    step_voxel_y = max(1, int(np.round(step_mm / spacing[1])))
    step_voxel_z = max(1, int(np.round(step_mm / spacing[2])))

    # 生成 grid 点索引
    xs = np.arange(0, sx, step_voxel_x)
    ys = np.arange(0, sy, step_voxel_y)
    zs = np.arange(0, sz, step_voxel_z)

    selected = []
    for z in zs:
        for y in ys:
            for x in xs:
                if mask[z, y, x]:
                    selected.append((z, y, x))

    print("Grid 排布晶格点数量:", len(selected))
    return selected


# -------------------------------
# 主流程
# -------------------------------
if __name__ == "__main__":
    output_dir = r"D:\debug\lattice"
    os.makedirs(output_dir, exist_ok=True)

    radius_mm = 5.0

    # 1. 创建手动 GTV
    gtv_img = create_manual_gtv(shape=(256, 256, 256), spacing=(1.0, 1.0, 1.0))
    sitk.WriteImage(gtv_img, os.path.join(output_dir, "gtv.nii.gz"))

    # 2. 生成晶格中心点
    vertices = generate_lattice_points_grid(gtv_img, radius=radius_mm, min_edge_dist_mm=5.0,
                                            shrink_mm=10.0)
    if len(vertices) == 0:
        exit("没有找到晶格点！")

    # 计算两两距离
    dist_matrix = compute_lattice_distances(vertices, gtv_img.GetSpacing())
    dm = dist_matrix.copy()
    np.fill_diagonal(dm, np.inf)

    min_dist_each = np.min(dm, axis=1)
    print("每个点的最近邻距离：", min_dist_each)
    print("所有晶格点最小最近邻距离：", np.min(min_dist_each))
    print("所有晶格点最大最近邻距离：", np.max(min_dist_each))

    print("晶格点数量:", len(vertices))
    # for v in vertices:
    #     print(v)

    # 3. 生成每个球 mask 并合并
    gtv_arr = sitk.GetArrayFromImage(gtv_img)
    merged = np.zeros_like(gtv_arr, dtype=np.uint8)

    for center in vertices:
        sphere_img = create_sphere_mask(center, radius_mm=radius_mm, ref_img=gtv_img)
        sphere_arr = sitk.GetArrayFromImage(sphere_img)
        # 截断到 GTV 范围
        sphere_arr = np.minimum(sphere_arr, gtv_arr)
        merged = np.maximum(merged, sphere_arr)

    lattice_img = sitk.GetImageFromArray(merged)
    lattice_img.CopyInformation(gtv_img)
    sitk.WriteImage(lattice_img, os.path.join(output_dir, "lattice_all.nii.gz"))
    print("晶格 mask 已保存:", os.path.join(output_dir, "lattice_all.nii.gz"))
