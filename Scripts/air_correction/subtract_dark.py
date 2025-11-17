import os
import numpy as np
from Scripts.air_correction.correct_image import read_dark_image
from SymmetryEstimation.utils import read_raw_image


def save_raw(path, array):
    array.astype(np.uint16).tofile(path)


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0709"
    image_size = 1420

    dark_path = r"D:\Data\cbct\air_correction\dark\dark_a.raw"
    dark = read_dark_image(dark_path, image_size)

    out_dir = data_dir + "_corrected"
    os.makedirs(out_dir, exist_ok=True)

    filenames = sorted([f for f in os.listdir(data_dir) if f.endswith(".raw")])

    for idx, fname in enumerate(filenames):
        raw_path = os.path.join(data_dir, fname)

        # 1. 读取原图
        image = read_raw_image(raw_path, image_size, image_size)

        # 2. 暗场扣除
        corrected = image - dark

        # 3. clip 避免负数
        corrected = np.clip(corrected, 0, None)

        # 4. 保存为新的 raw
        save_path = os.path.join(out_dir, fname)
        save_raw(save_path, corrected)

        print(f"[{idx + 1}/{len(filenames)}] Saved:", save_path)
