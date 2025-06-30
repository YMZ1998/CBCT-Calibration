import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


class DraggablePoints:
    """Draggable points on matplotlib figure."""

    def __init__(self, artists, tolerance=30):
        for artist in artists:
            artist.set_picker(tolerance)

        self.artists = artists
        self.final_coord = None

        self.fig = self.artists[0].figure
        self.ax = self.artists[0].axes

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_pressed)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.currently_dragging = False
        self.offset = np.zeros((1, 2))
        self.x0 = 0
        self.y0 = 0

        plt.title("Drag&Drop red points on the image.\nPress Enter to continue")
        plt.show()

    def on_press(self, event):
        isonartist = False
        for artist in self.artists:
            if artist.contains(event)[0]:
                isonartist = artist
        self.x0 = event.xdata
        self.y0 = event.ydata
        if isonartist:
            self.currently_dragging = True
            artist_center = np.array([a.center for a in self.artists])
            event_center = np.array([event.xdata, event.ydata])
            self.offset = artist_center - event_center

    def on_release(self, event):
        if self.currently_dragging:
            self.currently_dragging = False

    def on_motion(self, event):
        if self.currently_dragging:
            if event.xdata is None or event.ydata is None:
                print("Warning: Mouse pointer outside axes area. Cannot drag points here.")
                return  # 鼠标位置无效时不处理

            newcenters = np.array([event.xdata, event.ydata]) + self.offset

            img_dim = [self.ax.get_xlim()[1], self.ax.get_ylim()[0]]  # 根据坐标轴范围

            # 限制边界
            clipped_x = np.clip(newcenters[:, 0], 0, img_dim[0])
            clipped_y = np.clip(newcenters[:, 1], 0, img_dim[1])
            if not np.array_equal(newcenters[:, 0], clipped_x) or not np.array_equal(newcenters[:, 1], clipped_y):
                print("Warning: Points cannot be moved outside image boundaries.")
            newcenters[:, 0] = clipped_x
            newcenters[:, 1] = clipped_y

            for i, artist in enumerate(self.artists):
                artist.center = newcenters[i]
            self.fig.canvas.draw_idle()

    def on_key_pressed(self, event):
        step = 5  # 移动步长，可调节
        dx, dy = 0, 0
        if event.key == "left":
            dx = -step
        elif event.key == "right":
            dx = step
        elif event.key == "down":
            dy = step
        elif event.key == "up":
            dy = -step
        elif event.key == "enter":
            plt.close()
            return

        # 只有当不是拖动状态时，才响应键盘微调
        if not self.currently_dragging and (dx != 0 or dy != 0):
            for artist in self.artists:
                cx, cy = artist.center
                artist.center = (cx + dx, cy + dy)
            self.fig.canvas.draw_idle()

    def on_close(self, event):
        self.final_coord = self.get_coord()

    def get_coord(self):
        return np.array([a.center for a in self.artists])


def drag_and_drop_bbs(projection_img, bbs_projected, grayscale_range):
    """Manual correction of BBs using keyboard."""
    fig, ax = plt.subplots()
    ax.imshow(projection_img, cmap='gray', vmin=grayscale_range[0], vmax=grayscale_range[1])

    pts = []
    for x, y in bbs_projected:
        circle = patches.Circle((x, y), radius=5, color='r', alpha=0.6)
        ax.add_patch(circle)
        pts.append(circle)

    handler = DraggablePoints(pts)
    plt.show()  # 阻塞，等待用户关闭窗口
    return np.array(handler.final_coord)


# 模拟图像与钢珠坐标
img = np.full((512, 512), 100, dtype=np.uint16)

bbs = np.array([[100, 100], [200, 200], [300, 300]])

corrected = drag_and_drop_bbs(img, bbs, [0, 160])
print("Corrected:", corrected)
