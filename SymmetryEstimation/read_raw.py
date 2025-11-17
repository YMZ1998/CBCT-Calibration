import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from SymmetryEstimation.utils import read_raw_image

if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\CBCT0709" + "_corrected"
    image_size = 1420
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".raw")][::5]

    fig, ax = plt.subplots(figsize=(6, 6))
    img_handle = ax.imshow(np.zeros((image_size, image_size, 3), dtype=np.uint8))
    ax.axis("off")
    title_handle = ax.set_title("")

    for idx, fname in enumerate(sorted(filenames)):
        filename = os.path.join(data_dir, fname)
        print(f"🟡 正在处理第 {idx + 1} 张图像: {filename}")
        image = read_raw_image(filename, image_size, image_size)

        # image = np.clip(image, 500, 1500)
        min_val = np.min(image)
        max_val = np.max(image)

        norm = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        output = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        img_handle.set_data(output[..., ::-1])  # BGR -> RGB for matplotlib
        title_handle.set_text(f"Image: {fname}")
        plt.pause(0.1)

