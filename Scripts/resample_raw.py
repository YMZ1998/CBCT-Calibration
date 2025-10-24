import numpy as np
from scipy.ndimage import zoom

from SymmetryEstimation.utils import read_raw_image


def downsample_image(image, scale_factor, mode='mean'):
    if mode == 'mean':
        downsampled_image = zoom(image, scale_factor, order=1)
    elif mode == 'max':
        from skimage.measure import block_reduce
        downsampled_image = block_reduce(image, block_size=(int(1 / scale_factor), int(1 / scale_factor)), func=np.max)
    else:
        raise ValueError("不支持的 mode 参数，请选择 'mean' 或 'max'")

    return downsampled_image.astype(image.dtype)


def save_raw_image(image, filename):

    # 将图像数据转换为二进制并保存
    with open(filename, 'wb') as f:
        f.write(image.tobytes())


if __name__ == "__main__":
    filename = r"D:\\Data\\cbct\\Data\\ct_A002.raw"

    width, height = 4260, 4260
    image = read_raw_image(filename, width, height, dtype=np.uint16)

    scale_factor = 1 / 3
    downsampled_image = downsample_image(image, scale_factor, mode='mean')

    save_raw_image(downsampled_image, r"D:\\Data\\cbct\\Data\\ct_A002_1.raw")
