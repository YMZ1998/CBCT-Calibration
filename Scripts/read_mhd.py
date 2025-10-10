import SimpleITK as sitk
import matplotlib.pyplot as plt

# 设置 MHD 文件路径
# mhd_file_path = r'D:\Data\reg2d3d\ct2\ct.mhd'  # 替换为你的 MHD 文件路径
mhd_file_path = r'D:\Data\reg2d3d\dr2\drra.mhd'  # 替换为你的 MHD 文件路径

# 使用 SimpleITK 读取 MHD 文件
image = sitk.ReadImage(mhd_file_path)

# 打印图像的基本信息
print("Image Size:", image.GetSize())  # 图像尺寸 (x, y, z)
print("Pixel Type:", sitk.GetPixelIDValueAsString(image.GetPixelID()))  # 像素类型
print("Spacing:", image.GetSpacing())  # 图像的空间分辨率
print("Origin:", image.GetOrigin())  # 图像的原点偏移
print("Direction:", image.GetDirection())  # 图像的方向矩阵

# 将图像转换为 NumPy 数组
image_array = sitk.GetArrayFromImage(image)  # NumPy 数组，维度顺序为 (z, y, x)

# 打印 NumPy 数组的形状
print("Numpy Array Shape:", image_array.shape)  # 形状是 (z, y, x)

if len(image_array.shape)==3:
    # 显示某一层切片 (z-slice) 的图像
    slice_index = image_array.shape[0] // 2  # 取中间一层切片
    plt.imshow(image_array[slice_index, :, :], cmap='gray')
    plt.title(f"Slice at z={slice_index}")
else:
    plt.imshow(image_array, cmap='gray')

plt.show()
