import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from SymmetryEstimation.utils import read_raw_image


def detect_circles(image):
    image = np.clip(image, 500, 1500)
    min_val = np.min(image)
    max_val = np.max(image)

    norm = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=15, maxRadius=25)
    output = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), -1)
    return circles, output, cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    data_dir = r"D:\Data\cbct\体模"
    image_size = 1420
    filenames = [f for f in os.listdir(data_dir) if f.endswith(".raw")][::5]

    fig, ax = plt.subplots(figsize=(6, 6))
    img_handle = ax.imshow(np.zeros((image_size, image_size, 3), dtype=np.uint8))
    ax.axis("off")
    title_handle = ax.set_title("")

    frames = []  # 用于存储GIF的每帧

    for idx, fname in enumerate(sorted(filenames)):
        filename = os.path.join(data_dir, fname)
        print(f"\n🟡 正在处理第 {idx + 1} 张图像: {filename}")
        image = read_raw_image(filename, image_size, image_size)
        circles, output,_ = detect_circles(image)

        if circles is not None:
            print(f"✅ 检测到 {len(circles[0])} 个圆点：")
            for i, (x, y, r) in enumerate(circles[0]):
                print(f"    点 {i + 1}: x={x}, y={y}, r={r}")
        else:
            print("⚠️ 未检测到圆")

        img_handle.set_data(output[..., ::-1])  # BGR -> RGB for matplotlib
        title_handle.set_text(f"Detected Circles: {fname}")
        plt.pause(0.1)

        # 保存当前帧，转换为RGB格式
        frame_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    scale_factor = 0.25
    small_frames = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        new_size = (int(pil_img.width * scale_factor), int(pil_img.height * scale_factor))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
        small_frames.append(pil_img)

    # 保存 GIF，PIL 会自动优化调色板，文件通常会更小
    gif_path = os.path.join("../result", "detected_circles_compressed.gif")
    small_frames[0].save(
        gif_path,
        save_all=True,
        append_images=small_frames[1:],
        duration=100,  # 0.1秒
        loop=0,
        optimize=True,  # 开启优化调色板，减小文件
    )
    # # 循环结束后保存为GIF
    # import imageio
    # gif_path = os.path.join("./result", "detected_circles.gif")
    # imageio.mimsave(gif_path, frames, duration=0.1, loop=0)  # duration单位秒
    #
    # print(f"\n🎉 GIF保存成功: {gif_path}")
