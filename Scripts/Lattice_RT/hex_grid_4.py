import matplotlib.pyplot as plt
import numpy as np


def plot_hex_grid_3d(centers: np.ndarray, radius, nz=3):
    """
    Plot 3D hexagonal grid given XY center coordinates, radius, and number of layers in Z.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 六边形顶点角度
    # angles = np.array([np.pi / 6, np.pi / 2, 5 * np.pi / 6, 7 * np.pi / 6, 3 * np.pi / 2, 11 * np.pi / 6])
    angles = np.linspace(0 + 1 / 6 * np.pi, 2 * np.pi + 1 / 6 * np.pi, 6, endpoint=False)


    # 对每一层叠加Z坐标
    for iz in range(nz):
        z = iz * radius
        for x, y in centers:
            vx = x + radius * np.cos(angles)
            vy = y + radius * np.sin(angles)
            vz = np.full_like(vx, z)

            ax.scatter(vx, vy, vz, color='r', s=10)
            # 画六边形边
            for i in range(len(vx)):
                ax.plot([vx[i], vx[(i + 1) % 6]], [vy[i], vy[(i + 1) % 6]], [vz[i], vz[(i + 1) % 6]], 'k-',
                        linewidth=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 0.5])
    plt.show()


# ================= Example =================
def create_hex_grid_centers(nx: int = 5, ny: int = 5, radius: float = 1.) -> np.ndarray:
    """
    Create 2D hexagonal grid centers, can be stacked in Z later.
    """
    min_diam = radius * np.sqrt(3)
    ratio = np.sqrt(3) / 2

    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    gx = gx.astype(float)
    gx[1::2, :] += 0.5
    gy = gy.astype(float) * ratio

    gx = gx.reshape(-1, 1) * min_diam
    gy = gy.reshape(-1, 1) * min_diam
    centers = np.hstack([gx, gy])
    return centers


if __name__ == "__main__":
    radius = 2
    centers = create_hex_grid_centers(5, 5, radius)
    plot_hex_grid_3d(centers, radius, nz=5)
