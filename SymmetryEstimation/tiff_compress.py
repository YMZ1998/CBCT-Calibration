import scipy.ndimage
import tifffile

# 读取图像数据
input_file = 'D:\code\BM4D-GPU\data\salesman_noisy.tiff'
image_data = tifffile.imread(input_file)

# 下采样图像，按比例减小
downsampled_image = scipy.ndimage.zoom(image_data, (0.25, 0.25, 0.5))  # 将所有维度缩小为原来的一半

# 保存压缩后的图像
output_file = 'D:\code\BM4D-GPU\data\salesman_noisy2.tiff'
tifffile.imwrite(output_file, downsampled_image, compression='zlib')

print(f"下采样并压缩后的图像已保存为：{output_file}")
