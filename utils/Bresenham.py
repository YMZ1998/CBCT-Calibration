import matplotlib.pyplot as plt
import matplotlib.animation as animation

def bresenham(x0, y0, x1, y1):
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

# 设置起止点
x0, y0 = 2, 2
x1, y1 = 15, 8

points = bresenham(x0, y0, x1, y1)

# 准备画布
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(min(x0, x1) - 2, max(x0, x1) + 2)
ax.set_ylim(min(y0, y1) - 2, max(y0, y1) + 2)
ax.set_xticks(range(min(x0, x1) - 2, max(x0, x1) + 3))
ax.set_yticks(range(min(y0, y1) - 2, max(y0, y1) + 3))
ax.set_aspect('equal')
ax.grid(True)
ax.set_title(f"Bresenham Animation: ({x0},{y0}) → ({x1},{y1})")

line, = ax.plot([], [], 'ro-')

def update(frame):
    current_points = points[:frame+1]
    x_vals, y_vals = zip(*current_points)
    line.set_data(x_vals, y_vals)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(points), interval=300, blit=True, repeat=False)

plt.show()
