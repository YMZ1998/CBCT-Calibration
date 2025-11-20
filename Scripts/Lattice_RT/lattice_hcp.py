import os

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt

from Scripts.Lattice_RT.create_sphere_mask import create_sphere_mask
from Scripts.Lattice_RT.lattice_distance import create_manual_gtv, compute_lattice_distances


def generate_lattice_points_hcp(gtv_img, radius_mm=10.0, min_edge_dist_mm=2.0, shrink_mm=0.0):
    gtv = sitk.GetArrayFromImage(gtv_img).astype(np.uint8)  # z,y,x
    spacing = np.array(gtv_img.GetSpacing())  # (sx,sy,sz)
    origin = np.array(gtv_img.GetOrigin())
    direction = np.array(gtv_img.GetDirection()).reshape(3, 3)
    nz, ny, nx = gtv.shape

    # EDT in mm (note sampling order z,y,x -> spacing[z],spacing[y],spacing[x])
    dt = distance_transform_edt(gtv, sampling=(spacing[2], spacing[1], spacing[0]))
    gtv = dt >= shrink_mm + radius_mm
    # 基本 hex spacing in mm
    a = 2 * radius_mm + min_edge_dist_mm
    vx = a
    vy = np.sqrt(3) / 2 * a

    # bounding box (voxel indices come as z,y,x)
    coords = np.argwhere(gtv > 0)
    if coords.size == 0:
        return []
    zmin_v, ymin_v, xmin_v = coords.min(axis=0)
    zmax_v, ymax_v, xmax_v = coords.max(axis=0)

    # voxel->world (ix,iy,iz) -> world mm
    def voxel_to_world(ix, iy, iz):
        idx = np.array([ix + 0.5, iy + 0.5, iz + 0.5])
        # spacing order is x,y,z, and idx is [ix,iy,iz]
        return origin + direction.dot(idx * spacing)

    # world bbox corners
    p_min = voxel_to_world(xmin_v, ymin_v, zmin_v)
    p_max = voxel_to_world(xmax_v, ymax_v, zmax_v)
    x_min_world, y_min_world = min(p_min[0], p_max[0]) - a, min(p_min[1], p_max[1]) - a
    x_max_world, y_max_world = max(p_min[0], p_max[0]) + a, max(p_min[1], p_max[1]) + a

    # generate hex grid (full tight hex lattice) relative to origin_xy
    origin_xy = np.array([x_min_world, y_min_world])
    # compute how many cols/rows are needed to cover bbox
    n_cols = int(np.ceil((x_max_world - x_min_world) / vx)) + 4
    n_rows = int(np.ceil((y_max_world - y_min_world) / vy)) + 4

    pts_xy = []
    for row in range(-2, n_rows):
        y = row * vy
        x_offset = (vx / 2.0) if (row % 2 == 1) else 0.0
        for col in range(-2, n_cols):
            x = col * vx + x_offset
            w = origin_xy + np.array([x, y])
            # filter to bbox (slightly expanded)
            if (w[0] < x_min_world - a) or (w[0] > x_max_world + a) or (w[1] < y_min_world - a) or (
                w[1] > y_max_world + a):
                continue
            pts_xy.append(w)
    pts_xy = np.array(pts_xy)  # world coords

    # Z layers: use layer gap and alternate XY offset to create HCP-like stacking
    # For HCP we typically offset alternating layers by (a/2, sqrt(3)/6*a)
    layer_gap_mm = a  # 你可以调成 a * 0.816... 等更接近紧密堆叠的值
    z_steps = int(max(1, np.round((zmax_v - zmin_v + 1) * spacing[2] / layer_gap_mm)))  # rough count
    z_world_min = voxel_to_world(0, 0, zmin_v)[2]
    z_world_max = voxel_to_world(0, 0, zmax_v)[2]
    if z_world_max <= z_world_min:
        z_vals = [z_world_min]
    else:
        z_vals = np.arange(z_world_min, z_world_max + 1e-6, layer_gap_mm)

    vox_set = set()
    vox_list = []

    # helper: world point -> nearest voxel index
    def phys_to_index(pt_mm):
        rel = np.linalg.inv(direction).dot(pt_mm - origin) / spacing
        ix, iy, iz = np.round(rel).astype(int)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        iz = np.clip(iz, 0, nz - 1)
        return ix, iy, iz

    for li, wz in enumerate(z_vals):
        # alternating offset per layer for HCP
        if (li % 2) == 1:
            layer_offset = np.array([a / 2.0, np.sqrt(3) / 6.0 * a])
        else:
            layer_offset = np.array([0.0, 0.0])

        for wxy in pts_xy:
            pt_world = np.array([wxy[0], wxy[1], wz]) + np.append(layer_offset, 0.0)[:3]
            ix, iy, iz = phys_to_index(pt_world)
            key = (iz, iy, ix)
            if key in vox_set:
                continue
            if gtv[iz, iy, ix] == 0:
                continue
            if dt[iz, iy, ix] < radius_mm:
                continue
            vox_set.add(key)
            vox_list.append(key)

    return vox_list


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
    vertices = generate_lattice_points_hcp(gtv_img, radius_mm=radius_mm,
                                           min_edge_dist_mm=15.0, shrink_mm=10.0)

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
