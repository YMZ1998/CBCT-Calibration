import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

gif_path = '../result/detected_circles_compressed.gif'

# === 加载 GIF 并播放 ===
# 读取 GIF 为帧序列
im = Image.open(gif_path)
frames = []

try:
    while True:
        frames.append(np.array(im.copy()))
        im.seek(im.tell() + 1)
except EOFError:
    pass

# 显示为动画
fig, ax = plt.subplots()
img_display = ax.imshow(frames[0], cmap='magma')
ax.axis('off')


def update(frame):
    img_display.set_data(frame)
    return [img_display]


ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
plt.show()
