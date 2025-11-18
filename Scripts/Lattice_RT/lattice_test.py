import os

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi
from skimage.transform import resize

from Scripts.Lattice_RT.create_sphere_mask import create_sphere_mask


# -------------------------------
# 手动创建 GTV mask
# -------------------------------
def create_manual_gtv(shape=(64, 64, 64), spacing=(2.0, 2.0, 2.0), type='sphere'):
    sz, sy, sx = shape
    arr = np.zeros(shape, dtype=np.uint8)

    if type == 'sphere':
        center = np.array([sz // 2, sy // 2, sx // 2])
        radius = min(sz, sy, sx) // 4
        zz, yy, xx = np.meshgrid(np.arange(sz), np.arange(sy), np.arange(sx), indexing='ij')
        dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
        arr[dist2 <= radius ** 2] = 1
    elif type == 'cube':
        arr[sz // 4:3 * sz // 4, sy // 4:3 * sy // 4, sx // 4:3 * sx // 4] = 1

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


# -------------------------------
# 生成晶格点
# -------------------------------
def generate_lattice_points(gtv_img, radius_mm=10.0, min_edge_dist_mm=2.0, max_vertices=50):
    gtv = sitk.GetArrayFromImage(gtv_img).astype(np.uint8)
    spacing = gtv_img.GetSpacing()
    sz, sy, sx = gtv.shape

    # dt = distance_transform_edt(gtv, sampling=(spacing[2], spacing[1], spacing[0]))

    # 下采样
    factor = 4  # 每个维度下采样2倍
    gtv_small = gtv[::factor, ::factor, ::factor]

    # 距离变换
    dt_small = ndi.distance_transform_edt(gtv_small, sampling=(spacing[2], spacing[1], spacing[0]))

    # 放大回原始大小
    dt = resize(dt_small, gtv.shape, order=1, mode='edge', anti_aliasing=False)
    dt = dt * factor  # 物理尺寸也需要乘回

    # dt_img = sitk.SignedMaurerDistanceMap(gtv_img,
    #                                       insideIsPositive=True,
    #                                       useImageSpacing=True)
    # dt = sitk.GetArrayFromImage(dt_img)

    # 筛选候选点：球完全在 mask 内
    candidates = []
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                if gtv[z, y, x] == 0:
                    continue
                if dt[z, y, x] < radius_mm:  # 球心到边界 < 半径
                    continue
                candidates.append((z, y, x))
    print("候选点数量:", len(candidates))

    # 贪心选择：球间距
    selected = []
    rvz, rvy, rvx = (radius_mm * 2 + min_edge_dist_mm) / spacing[2], (radius_mm * 2 + min_edge_dist_mm) / spacing[1], (
        radius_mm * 2 + min_edge_dist_mm) / spacing[0]

    for cz, cy, cx in candidates[::100]:
        ok = True
        for sz_, sy_, sx_ in selected:
            dz_mm = (cz - sz_) * spacing[2]
            dy_mm = (cy - sy_) * spacing[1]
            dx_mm = (cx - sx_) * spacing[0]
            dist = np.sqrt(dx_mm ** 2 + dy_mm ** 2 + dz_mm ** 2)
            if dist < radius_mm * 2 + min_edge_dist_mm:
                ok = False
                break
        if ok:
            selected.append((cz, cy, cx))
            # if len(selected) >= max_vertices:
            #     break
    print("已选择晶格点数量:", len(selected))
    return selected


# -------------------------------
# 主流程
# -------------------------------
if __name__ == "__main__":
    output_dir = r"D:\debug\lattice"
    os.makedirs(output_dir, exist_ok=True)

    radius_mm = 20.0

    # 1. 创建手动 GTV
    gtv_img = create_manual_gtv(shape=(256, 256, 256), spacing=(1.0, 1.0, 2.0), type='sphere')
    sitk.WriteImage(gtv_img, os.path.join(output_dir, "gtv.nii.gz"))

    # 2. 生成晶格中心点
    vertices = generate_lattice_points(gtv_img, radius_mm=radius_mm, min_edge_dist_mm=20.0, max_vertices=20)
    print("晶格点数量:", len(vertices))
    for v in vertices:
        print(v)

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
