import SimpleITK as sitk
import numpy as np
from bm4d import bm4d, BM4DStages, BM4DProfile


def split_volume(vol: np.ndarray, block_size: int = 128, overlap: int = 16):
    """
    Split a 3D volume into overlapping blocks for BM4D processing.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (Z, Y, X)
    block_size : int
        Maximum size of each block
    overlap : int
        Number of overlapping voxels between blocks

    Yields
    ------
    coords : tuple
        (z0, z1, y0, y1, x0, x1)
    block : np.ndarray
        The sub-volume
    """
    z_max, y_max, x_max = vol.shape
    step = block_size - overlap
    for z0 in range(0, z_max, step):
        for y0 in range(0, y_max, step):
            for x0 in range(0, x_max, step):
                z1 = min(z0 + block_size, z_max)
                y1 = min(y0 + block_size, y_max)
                x1 = min(x0 + block_size, x_max)
                yield (z0, z1, y0, y1, x0, x1), vol[z0:z1, y0:y1, x0:x1]


def denoise_nii(input_path: str,
                output_path: str,
                sigma: float = 25 / 255.0,
                normalize: bool = True,
                block_size: int = 128,
                overlap: int = 16):
    # ---- 1. Read image ----
    print(f"Reading: {input_path}")
    img_sitk = sitk.ReadImage(input_path)
    vol = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # shape: (Z, Y, X)

    # ---- 2. Normalize if needed ----
    if normalize:
        vmin, vmax = vol.min(), vol.max()
        vol_norm = (vol - vmin) / (vmax - vmin)
        sigma_used = sigma
        print(f"Data normalized to [0,1], sigma={sigma_used}")
    else:
        vol_norm = vol
        sigma_used = sigma
        print(f"No normalization, sigma={sigma_used}")

    # ---- 3. Prepare profile ----
    profile = BM4DProfile()
    profile.split_block_extraction = [False, True, True]

    # ---- 4. Denoise block by block ----
    denoised_vol = np.zeros_like(vol_norm, dtype=np.float32)
    weight_vol = np.zeros_like(vol_norm, dtype=np.float32)  # 用于边界加权

    print("Running BM4D block-wise denoising ...")
    for coords, block in split_volume(vol_norm, block_size=block_size, overlap=overlap):
        z0, z1, y0, y1, x0, x1 = coords
        denoised_block = bm4d(block,
                              sigma_psd=sigma_used,
                              profile=profile,
                              stage_arg=BM4DStages.ALL_STAGES)
        # 累加结果和权重
        denoised_vol[z0:z1, y0:y1, x0:x1] += denoised_block
        weight_vol[z0:z1, y0:y1, x0:x1] += 1.0

    # ---- 5. 平均重叠部分 ----
    denoised_vol /= weight_vol

    # ---- 6. Rescale back if normalized ----
    if normalize:
        denoised_vol = denoised_vol * (vmax - vmin) + vmin

    # ---- 7. Save output ----
    out_img = sitk.GetImageFromArray(denoised_vol)
    out_img.CopyInformation(img_sitk)
    sitk.WriteImage(out_img, output_path)
    print(f"Denoised volume saved to: {output_path}")


if __name__ == "__main__":
    input_file = r"D:\debug\A_output_1024.mhd"
    output_file = r"D:\debug\denoised.nii.gz"
    denoise_nii(input_file,
                output_file,
                sigma=25 / 255.0,
                normalize=True,
                block_size=128,
                overlap=16)
