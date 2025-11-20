import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def plot_hex_grid(centers: np.ndarray, min_diam: float = 1., plotting_gap: float = 0., show_vertices: bool = True):
    """
    Plot hexagonal grid given center coordinates and mark vertices.
    """
    radius = min_diam / np.sqrt(3) * (1 - plotting_gap)
    fig, ax = plt.subplots(figsize=(5, 5))

    angles = np.linspace(0 + 1 / 6 * np.pi, 2 * np.pi + 1 / 6 * np.pi, 6, endpoint=False)  # 六边形顶点角度
    print(angles)

    for x, y in centers:
        hexagon = mpatches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=radius,
            orientation=np.deg2rad(0),
            facecolor='none',
            edgecolor='k',
            linewidth=0.5
        )
        ax.add_patch(hexagon)

        if show_vertices:
            vx = x + radius * np.cos(angles)
            vy = y + radius * np.sin(angles)
            ax.scatter(vx, vy, color='r', s=10)

    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.show()


# ================= Example =================
if __name__ == "__main__":
    def create_hex_grid_centers(nx: int = 5, ny: int = 5, min_diam: float = 1.) -> np.ndarray:
        ratio = np.sqrt(3) / 2
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
        gx = gx.astype(float)
        gx[1::2, :] += 0.5
        gy = gy.astype(float) * ratio
        gx = gx.reshape(-1, 1) * min_diam
        gy = gy.reshape(-1, 1) * min_diam
        centers = np.hstack([gx, gy])
        return centers


    centers = create_hex_grid_centers()
    plot_hex_grid(centers, show_vertices=True)
