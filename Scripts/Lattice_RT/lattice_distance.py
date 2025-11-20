import os
import time

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi

from Scripts.Lattice_RT.create_manual_gtv import create_manual_gtv
from Scripts.Lattice_RT.create_sphere_mask import create_sphere_mask


def generate_lattice_points(gtv_img, radius=10.0, min_edge_dist_mm=2.0, shrink_mm=5.0):
    gtv = sitk.GetArrayFromImage(gtv_img).astype(np.uint8)
    spacing = gtv_img.GetSpacing()
    # from Scripts.Lattice_RT.shrink_mask import shrink_mask
    # gtv = shrink_mask(gtv, spacing[::-1], shrink_mm)

    start = time.time()
    # is_resample = 0
    # if is_resample:
    #     # 下采样
    #     factor = 4  # 每个维度下采样2倍
    #     gtv_small = gtv[::factor, ::factor, ::factor]
    #
    #     # 距离变换
    #     dt_small = ndi.distance_transform_edt(gtv_small, sampling=spacing[::-1])
    #
    #     # 放大回原始大小
    #     from skimage.transform import resize
    #     dt = resize(dt_small, gtv.shape, order=1, mode='edge', anti_aliasing=False)
    #     dt = dt * factor  # 物理尺寸也需要乘回
    # else:
    dt = ndi.distance_transform_edt(gtv, sampling=spacing[::-1])

    candidates = np.argwhere(dt >= radius + shrink_mm)

    print("候选点数量:", len(candidates))
    end = time.time()
    print("select candidates 耗时: {:.3f} 秒".format(end - start))
    # 贪心选择：球间距
    selected = []

    for cz, cy, cx in candidates[::]:
        ok = True
        for sz_, sy_, sx_ in selected:
            dz_mm = (cz - sz_) * spacing[2]
            dy_mm = (cy - sy_) * spacing[1]
            dx_mm = (cx - sx_) * spacing[0]
            dist = np.sqrt(dx_mm ** 2 + dy_mm ** 2 + dz_mm ** 2)
            if dist < radius * 2 + min_edge_dist_mm:
                ok = False
                break
        if ok:
            selected.append((cz, cy, cx))
            # if len(selected) >= max_vertices:
            #     break
    print("已选择晶格点数量:", len(selected))
    return selected


def compute_lattice_distances(selected, spacing):
    n = len(selected)
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        z1, y1, x1 = selected[i]
        for j in range(i + 1, n):
            z2, y2, x2 = selected[j]

            dz = (z1 - z2) * spacing[2]
            dy = (y1 - y2) * spacing[1]
            dx = (x1 - x2) * spacing[0]

            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


if __name__ == "__main__":
    output_dir = r"D:\debug\lattice"
    os.makedirs(output_dir, exist_ok=True)

    radius_mm = 18.0

    # 1. 创建手动 GTV
    gtv_img = create_manual_gtv(shape=(256, 256, 256), spacing=(1.0, 1.0, 1.0))
    sitk.WriteImage(gtv_img, os.path.join(output_dir, "gtv.nii.gz"))

    # 2. 生成晶格中心点
    vertices = generate_lattice_points(gtv_img, radius=radius_mm, min_edge_dist_mm=20.0,
                                       shrink_mm=10.0)

    # 计算两两距离
    dist_matrix = compute_lattice_distances(vertices, gtv_img.GetSpacing())
    # print(dist_matrix)
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
