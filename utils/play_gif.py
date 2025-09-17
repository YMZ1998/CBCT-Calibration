import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

gif_path = '../result/detected_circles_compressed.gif'

im = Image.open(gif_path)
frames = []

try:
    while True:
        frames.append(np.array(im.copy()))
        im.seek(im.tell() + 1)
except EOFError:
    pass

fig, ax = plt.subplots()
img_display = ax.imshow(frames[0], cmap='magma')
ax.axis('off')


def update(frame):
    img_display.set_data(frame)
    return [img_display]


ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
plt.show()
