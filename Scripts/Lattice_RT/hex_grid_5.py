import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# 假设 vertices 是 (N,6,3) 的 numpy 数组
# reshape 成 (N*6,3) 点云
def save_hex_grid_point_cloud(vertices, filename="hex_grid.ply"):
    points = vertices.reshape(-1, 3)
    # 创建 open3d 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 保存为 ply
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {points.shape[0]} points to {filename}")


def create_hex_grid_centers_and_vertices(size, spacing, origin, radius):
    """
    Create 3D hexagonal grid centers and vertices in physical coordinates.

    Returns:
        centers: (N, 3) array of hexagon centers
        vertices: (N, 6, 3) array of 6 vertices per hexagon
    """
    width, height, depth = size
    sx, sy, sz = spacing
    ox, oy, oz = origin

    # 六边形间距
    dx = radius * np.sqrt(3) * sx
    dy = radius * 1.5 * sy
    dz = radius * sz

    centers = []
    vertices = []

    row = 0
    y = 0
    while y <= height:
        x_offset = 0.5 * dx if row % 2 == 1 else 0
        x = x_offset
        while x <= width:
            z = 0
            while z <= depth:
                center = np.array([ox + x, oy + y, oz + z])
                centers.append(center)

                # 六边形顶点
                angles = np.array([np.pi / 6, np.pi / 2, 5 * np.pi / 6, 7 * np.pi / 6, 3 * np.pi / 2, 11 * np.pi / 6])
                vx = center[0] + radius * sx * np.cos(angles)
                vy = center[1] + radius * sy * np.sin(angles)
                vz = np.full_like(vx, center[2])
                verts = np.stack([vx, vy, vz], axis=1)
                vertices.append(verts)

                z += dz
            x += dx
        row += 1
        y += dy
    vertices = np.array(vertices)
    vertices = vertices.reshape(-1, 3)
    unique_vertices = np.unique(vertices, axis=0)
    return np.array(centers), unique_vertices  # centers (N,3), vertices (N,6,3)


def plot_hex_grid_3d(centers, vertices):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制中心点
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='b', s=5)

    # 绘制六边形顶点和边
    for verts in vertices:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], color='r', s=10)
        for i in range(6):
            ax.plot([verts[i, 0], verts[(i + 1) % 6, 0]],
                    [verts[i, 1], verts[(i + 1) % 6, 1]],
                    [verts[i, 2], verts[(i + 1) % 6, 2]], 'k-', lw=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    plt.show()


if __name__ == "__main__":
    size = (100, 100, 100)
    radius = 20.0
    spacing = (1.0, 1., 1.5)
    origin = (0.0, 0.0, 0.0)

    centers, vertices = create_hex_grid_centers_and_vertices(size, spacing, origin, radius)

    print("Hexagon centers:", centers.shape, len(centers) * 6)
    print("Hexagon vertices:", vertices.shape)
    # plot_hex_grid_3d(centers, vertices)
    save_hex_grid_point_cloud(vertices, r"D:\debug\lattice\hex_grid.ply")
