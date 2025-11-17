import os

import SimpleITK as sitk
import numpy as np



def create_sphere_mask(center_zyx, radius_mm, ref_img):
    """
    使用迭代方式生成球形 mask（物理空间为球），不使用网格
    center_zyx: (z, y, x) 体素坐标
    radius_mm: 球半径 (mm)
    """
    spacing = ref_img.GetSpacing()  # (sx, sy, sz)
    size = ref_img.GetSize()       # (x, y, z)
    sz, sy, sx = ref_img.GetSize()[2], ref_img.GetSize()[1], ref_img.GetSize()[0]

    # 输出数组初始化
    mask = np.zeros((sz, sy, sx), dtype=np.uint8)

    cz, cy, cx = center_zyx

    # 半径转换为 voxel 数（各轴）
    radius_vox = np.array([
        radius_mm / spacing[2],  # z
        radius_mm / spacing[1],  # y
        radius_mm / spacing[0],  # x
    ])

    # 遍历体素范围（限定在球的包围盒内）
    z_min = max(int(cz - radius_vox[0]), 0)
    z_max = min(int(cz + radius_vox[0]) + 1, sz)
    y_min = max(int(cy - radius_vox[1]), 0)
    y_max = min(int(cy + radius_vox[1]) + 1, sy)
    x_min = max(int(cx - radius_vox[2]), 0)
    x_max = min(int(cx + radius_vox[2]) + 1, sx)

    for z in range(z_min, z_max):
        dz = (z - cz) / radius_vox[0]
        dz2 = dz * dz
        for y in range(y_min, y_max):
            dy = (y - cy) / radius_vox[1]
            dy2 = dy * dy
            for x in range(x_min, x_max):
                dx = (x - cx) / radius_vox[2]
                dx2 = dx * dx
                if dx2 + dy2 + dz2 <= 1.0:
                    mask[z, y, x] = 1
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(ref_img)
    return out


if __name__ == "__main__":
    gtv_path = r"D:\Data\cbct\CT2\skull_mask.mhd"
    output_dir = r"D:\debug\lattice"
    os.makedirs(output_dir, exist_ok=True)

    gtv_img = sitk.ReadImage(gtv_path)

    # 示例球中心坐标
    center_voxel = (21, 211, 215)
    sphere_img = create_sphere_mask(center_voxel, radius_mm=10.0, ref_img=gtv_img)

    out_path = os.path.join(output_dir, "sphere_img.nii.gz")
    sitk.WriteImage(sphere_img, out_path)
    print("保存完成:", out_path)
