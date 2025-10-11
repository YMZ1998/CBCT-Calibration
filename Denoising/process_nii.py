import SimpleITK as sitk
import cv2
import numpy as np

from Denoising.bilateral_raw import visualize_images
from Denoising.bm3d_raws import bm3d_denoise_raw
from Denoising.conditional_median_raws import conditional_median_raw

if __name__ == '__main__':
    case_path = r'D:\debug\cbct\A_output_1024_-0.67.mhd'
    image = sitk.ReadImage(case_path)
    image = sitk.GetArrayFromImage(image)
    print(image.shape)
    slice_index = image.shape[0] // 2
    # slice_index = 195
    slice = image[slice_index, :, :]
    denoised = conditional_median_raw(slice)
    # denoised = cv2.bilateralFilter(slice.astype(np.float32), d=9, sigmaColor=25, sigmaSpace=25).astype(np.int32)
    # denoised = bm3d_denoise_raw(slice, 40).astype(np.int32)

    a=200
    b=1000
    slice = np.clip(slice, a, b)
    denoised = np.clip(denoised, a, b)
    visualize_images(slice, denoised)

    denoised.tofile(r'D:\debug\cbct\denoised.raw')
