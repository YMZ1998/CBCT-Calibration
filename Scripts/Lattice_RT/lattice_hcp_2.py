import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from Scripts.Lattice_RT.create_sphere_mask import create_sphere_mask_physical
from Scripts.Lattice_RT.lattice_distance import create_manual_gtv, compute_lattice_distances


def create_hex_grid_centers_and_vertices(gtv_img, sphere_dist, sphere_radius, shrink_mm=10):
    width, height, depth = gtv_img.GetSize()
    spacing = gtv_img.GetSpacing()  # (sx,sy,sz)
    print(gtv_img.GetSize())
    sx, sy, sz = gtv_img.GetSpacing()
    ox, oy, oz = gtv_img.GetOrigin()

    # 六边形间距
    dx = sphere_dist * np.sqrt(3)
    dy = sphere_dist * 1.5
    dz = sphere_dist

    centers = []
    vertices = []

    # 转换为 numpy mask
    gtv_np = sitk.GetArrayFromImage(gtv_img)  # shape: [D, H, W]
    dt = distance_transform_edt(gtv_np, sampling=(spacing[2], spacing[1], spacing[0]))
    gtv_np = dt >= shrink_mm + sphere_radius

    row = 0
    y = 0
    while y <= height:
        x_offset = 0.5 * dx if row % 2 == 1 else 0
        x = x_offset
        while x <= width:
            z = 0
            while z <= depth:
                center = np.array([ox + x, oy + y, oz + z])
                # 检查中心点是否在 mask 内
                idx_x = int(round((center[0] - ox) / sx))
                idx_y = int(round((center[1] - oy) / sy))
                idx_z = int(round((center[2] - oz) / sz))
                if 0 <= idx_x < width and 0 <= idx_y < height and 0 <= idx_z < depth:
                    if gtv_np[idx_z, idx_y, idx_x]:
                        centers.append(center)

                        # 六边形顶点
                        angles = np.array([np.pi / 6, np.pi / 2, 5 * np.pi / 6,
                                           7 * np.pi / 6, 3 * np.pi / 2, 11 * np.pi / 6])
                        vx = center[0] + sphere_dist * sx * np.cos(angles)
                        vy = center[1] + sphere_dist * sy * np.sin(angles)
                        vz = np.full_like(vx, center[2])
                        verts = np.stack([vx, vy, vz], axis=1)

                        # 保留在 mask 内的顶点
                        mask = []
                        for v in verts:
                            ix = int(round((v[0] - ox) / sx))
                            iy = int(round((v[1] - oy) / sy))
                            iz = int(round((v[2] - oz) / sz))
                            if 0 <= ix < width and 0 <= iy < height and 0 <= iz < depth:
                                if gtv_np[iz, iy, ix]:
                                    mask.append(v)
                        if mask:
                            vertices.extend(mask)

                z += dz
            x += dx
        row += 1
        y += dy

    vertices_rounded = np.round(vertices, decimals=4)  # 保留 4 位小数，可根据体素尺寸调整
    vertices_unique = np.unique(vertices_rounded, axis=0)

    return np.array(centers), vertices_unique


def plot_hex_grid_3d(centers, vertices):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制中心点
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='b', s=5)

    # 绘制六边形顶点和边
    for verts in vertices:
        ax.scatter(verts[0], verts[1], verts[2], color='r', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    plt.show()


if __name__ == "__main__":
    output_dir = r"D:\debug\lattice"
    os.makedirs(output_dir, exist_ok=True)

    sphere_radius = 7.5
    sphere_dist = sphere_radius * 3
    print("球半径:", sphere_radius)
    print("六边形间距:", sphere_dist)
    shrink_mm = 10

    # 1. 创建手动 GTV
    gtv_img = create_manual_gtv(shape=(256, 256, 128), spacing=(1.0, 1.0, 2.0))
    sitk.WriteImage(gtv_img, os.path.join(output_dir, "gtv.nii.gz"))

    centers, vertices = create_hex_grid_centers_and_vertices(gtv_img, sphere_dist, sphere_radius, shrink_mm)
    print("Hexagon centers:", centers.shape, len(centers) * 6)
    print("Hexagon vertices:", vertices.shape)
    # plot_hex_grid_3d(centers, vertices)
    # save_hex_grid_point_cloud(vertices, r"D:\debug\lattice\hex_grid.ply")

    if len(vertices) == 0:
        exit("没有找到晶格点！")

    # 计算两两距离
    dist_matrix = compute_lattice_distances(vertices, (1, 1, 1))
    dm = dist_matrix.copy()
    np.fill_diagonal(dm, np.inf)

    min_dist_each = np.min(dm, axis=1)
    # print("每个点的最近邻距离：", min_dist_each)
    print("所有晶格点最小最近邻距离：", np.min(min_dist_each))
    print("所有晶格点最大最近邻距离：", np.max(min_dist_each))
    print("晶格点数量:", len(vertices))

    # 3. 生成每个球 mask 并合并
    gtv_arr = sitk.GetArrayFromImage(gtv_img)
    merged = np.zeros_like(gtv_arr, dtype=np.uint8)

    for center in vertices:
        sphere_img = create_sphere_mask_physical(center, radius_mm=sphere_radius, ref_img=gtv_img)
        sphere_arr = sitk.GetArrayFromImage(sphere_img)
        # 截断到 GTV 范围
        sphere_arr = np.minimum(sphere_arr, gtv_arr)
        merged = np.maximum(merged, sphere_arr)

    lattice_img = sitk.GetImageFromArray(merged)
    lattice_img.CopyInformation(gtv_img)
    sitk.WriteImage(lattice_img, os.path.join(output_dir, "lattice_all.nii.gz"))
    print("晶格 mask 已保存:", os.path.join(output_dir, "lattice_all.nii.gz"))
