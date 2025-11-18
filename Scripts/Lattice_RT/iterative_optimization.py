import os
import shutil
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from Scripts.Lattice_RT.create_sphere_mask import create_sphere_mask


# -----------------------------------------
# 动态迭代优化布置晶格点（无球体越界）
# -----------------------------------------
def iterative_optimization_lattice(gtv_img, radius_mm=15.0, min_edge_dist_mm=2.0,
                                   max_vertices=20, max_iter=200):

    gtv_arr = sitk.GetArrayFromImage(gtv_img)
    spacing = gtv_img.GetSpacing()
    sz, sy, sx = gtv_arr.shape

    # ---------- 计算距离场 ----------
    dt_mm = distance_transform_edt(gtv_arr, sampling=(spacing[2], spacing[1], spacing[0]))

    # ---------- 候选点：必须满足球体完全落入GTV ----------
    candidates = np.argwhere(dt_mm >= radius_mm)

    np.random.shuffle(candidates)
    selected = []

    # ---------- 初始点选择 ----------
    for pt in candidates:
        if len(selected) >= max_vertices:
            break
        z, y, x = pt

        ok = True
        for cz, cy, cx in selected:
            dz = (z - cz) * spacing[2]
            dy = (y - cy) * spacing[1]
            dx = (x - cx) * spacing[0]
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if dist < 2 * radius_mm + min_edge_dist_mm:
                ok = False
                break

        if ok:
            selected.append((z, y, x))

    selected = np.array(selected, dtype=int)

    # ---------- 迭代优化 ----------
    for it in range(max_iter):
        for i, (cz, cy, cx) in enumerate(selected):

            # 局部随机移动
            dz = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)
            dx = np.random.randint(-1, 2)
            nz, ny, nx = cz + dz, cy + dy, cx + dx

            # 越界检查
            if not (0 <= nz < sz and 0 <= ny < sy and 0 <= nx < sx):
                continue

            # 必须在GTV内部
            if gtv_arr[nz, ny, nx] == 0:
                continue

            # 必须距离足够大，保证球不会出界
            if dt_mm[nz, ny, nx] < radius_mm:
                continue

            # 检查与其他球中心的最小距离
            ok = True
            for j, (cz2, cy2, cx2) in enumerate(selected):
                if i == j:
                    continue
                dist = np.sqrt(((nz - cz2) * spacing[2])**2 +
                               ((ny - cy2) * spacing[1])**2 +
                               ((nx - cx2) * spacing[0])**2)
                if dist < 2 * radius_mm + min_edge_dist_mm:
                    ok = False
                    break

            if ok:
                selected[i] = (nz, ny, nx)

    return selected


# -----------------------------------------
# 主程序：生成不会越界的晶格球体
# -----------------------------------------
if __name__ == "__main__":
    gtv_path = r"D:\debug\gtv_manual.nii.gz"
    output_dir = r"D:\debug\lattice"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(gtv_path, os.path.join(output_dir, "gtv.nii.gz"))

    gtv_img = sitk.ReadImage(gtv_path)
    gtv_arr = sitk.GetArrayFromImage(gtv_img)

    radius_mm = 10.0

    # ------- 计算球心点位 -------
    vertices = iterative_optimization_lattice(gtv_img,
                                              radius_mm=radius_mm,
                                              min_edge_dist_mm=2.0,
                                              max_vertices=20,
                                              max_iter=300)

    print("最终晶格点数量:", len(vertices))
    for v in vertices:
        print(v)

    # ------- 生成球 mask -------
    merged = np.zeros_like(gtv_arr, dtype=np.uint8)

    for center in vertices:
        sphere_img = create_sphere_mask(center, radius_mm=radius_mm, ref_img=gtv_img)
        sphere_arr = sitk.GetArrayFromImage(sphere_img)

        # ★ 自动裁剪：防止越界（强制 sphere ∩ GTV）
        sphere_arr = np.logical_and(sphere_arr, gtv_arr).astype(np.uint8)

        merged = np.maximum(merged, sphere_arr)

    merged_img = sitk.GetImageFromArray(merged)
    merged_img.CopyInformation(gtv_img)

    out_path = os.path.join(output_dir, "lattice_all.nii.gz")
    sitk.WriteImage(merged_img, out_path)

    print("合并晶格 mask 已保存:", out_path)
