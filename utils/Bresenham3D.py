import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def bresenham_3d(x0, y0, z0, x1, y1, z1):
    """
    3D Bresenham algorithm to traverse voxels from (x0, y0, z0) to (x1, y1, z1)
    Returns list of (x, y, z) voxel indices
    """
    voxels = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    x, y, z = x0, y0, z0

    if dx >= dy and dx >= dz:  # x dominant
        yd = dy - dx // 2
        zd = dz - dx // 2
        for _ in range(dx + 1):
            voxels.append((x, y, z))
            if yd >= 0:
                y += sy
                yd -= dx
            if zd >= 0:
                z += sz
                zd -= dx
            x += sx
            yd += dy
            zd += dz
    elif dy >= dx and dy >= dz:  # y dominant
        xd = dx - dy // 2
        zd = dz - dy // 2
        for _ in range(dy + 1):
            voxels.append((x, y, z))
            if xd >= 0:
                x += sx
                xd -= dy
            if zd >= 0:
                z += sz
                zd -= dy
            y += sy
            xd += dx
            zd += dz
    else:  # z dominant
        xd = dx - dz // 2
        yd = dy - dz // 2
        for _ in range(dz + 1):
            voxels.append((x, y, z))
            if xd >= 0:
                x += sx
                xd -= dz
            if yd >= 0:
                y += sy
                yd -= dz
            z += sz
            xd += dx
            yd += dy

    return voxels


# 设置起止点
start = (2, 2, 1)
end = (10, 6, 8)
path = bresenham_3d(*start, *end)

# 创建画布
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_zlim(0, 15)
ax.set_xticks(range(0, 16, 1))
ax.set_yticks(range(0, 16, 1))
ax.set_zticks(range(0, 16, 1))
ax.set_title(f"Bresenham 3D Voxel Traversal\n{start} → {end}")
ax.view_init(elev=20, azim=45)
ax.grid(True)

voxels_plot = []


def update(frame):
    if frame < len(path):
        x, y, z = path[frame]
        voxel = ax.bar3d(x, y, z, 1, 1, 1, color='red', edgecolor='black', alpha=0.6)
        voxels_plot.append(voxel)
    return voxels_plot


ani = animation.FuncAnimation(fig, update, frames=len(path), interval=300, repeat=False)

plt.show()

points = bresenham_3d(*start, *end)
points = np.array(points)

# 3D 可视化
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o', label='Line Voxels')
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='lightgray', linewidth=1, alpha=0.6)

# 起点和终点高亮
ax.scatter(*start, c='green', s=100, label='Start', marker='o')
ax.scatter(*end, c='red', s=100, label='End', marker='^')

# 网格和标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Bresenham Voxel Line')
ax.legend()
ax.grid(True)
ax.view_init(elev=30, azim=135)  # 可旋转视角

plt.tight_layout()
plt.show()
