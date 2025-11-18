import time

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize

# 假设 gtv_img 已经读取
gtv_img = sitk.ReadImage(r"D:\debug\gtv_manual.nii.gz")
gtv = sitk.GetArrayFromImage(gtv_img).astype(np.uint8)
spacing = gtv_img.GetSpacing()

# -------------------------
# 测试 distance_transform_edt 耗时
# -------------------------
start = time.time()
dt = distance_transform_edt(gtv, sampling=(spacing[2], spacing[1], spacing[0]))
end = time.time()
print("distance_transform_edt 耗时: {:.3f} 秒".format(end - start))

start = time.time()

# 下采样
factor = 2  # 每个维度下采样2倍
gtv_small = gtv[::factor, ::factor, ::factor]

# 距离变换
dt_small = ndi.distance_transform_edt(gtv_small)

# 放大回原始大小
dt = resize(dt_small, gtv.shape, order=1, mode='edge', anti_aliasing=False)
dt = dt * factor  # 物理尺寸也需要乘回

end = time.time()
print("distance_transform_edt 耗时: {:.3f} 秒".format(end - start))

# -------------------------
# 测试腐蚀法耗时
# -------------------------
radius_mm = 20.0
radius_vox = int(radius_mm / min(spacing))
radius_list = [radius_vox] * gtv_img.GetDimension()

start = time.time()
gtv_eroded = sitk.BinaryErode(gtv_img, radius_list)
candidates = np.argwhere(sitk.GetArrayFromImage(gtv_eroded) > 0)
end = time.time()
print("腐蚀法生成候选点耗时: {:.3f} 秒".format(end - start))
print("候选点数量:", len(candidates))
