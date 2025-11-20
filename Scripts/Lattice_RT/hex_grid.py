# -----------------------------------------------------------
# hexalattice module creates and prints hexagonal lattices
#
# (C) 2020 Alex Kazakov,
# Released under MIT License
# email alex.kazakov@mail.huji.ac.il
# Full documentation: https://github.com/alexkaz2/hexalattice
# -----------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from typing import List, Union


# ============================================================
#                   Public API
# ============================================================

def create_hex_grid(
        nx: int = 4,
        ny: int = 5,
        min_diam: float = 1.,
        n: int = 0,
        align_to_origin: bool = True,
        face_color: Union[List[float], str] = None,
        edge_color: Union[List[float], str] = None,
        plotting_gap: float = 0.,
        crop_circ: float = 0.,
        do_plot: bool = False,
        rotate_deg: float = 0.,
        keep_x_sym: bool = True,
        h_ax: plt.Axes = None,
        line_width: float = 0.2,
        background_color: Union[List[float], str] = None
) -> (np.ndarray, plt.Axes):
    """
    Create hexagonal lattice center coordinates, and optionally plot them.

    Args:
        nx, ny: number of hexagons in X/Y grid.
        min_diam: minimal diameter of each hexagon.
        n: force total number of hexagons, overrides (nx, ny).
        align_to_origin: shift the grid so that the central tile is centered at origin.
        face_color, edge_color: polygon styles.
        plotting_gap: gap between hexagons (fraction of min_diam).
        crop_circ: if >0, keep only hexagons within circular radius (centered).
        do_plot: whether to draw the grid.
        rotate_deg: rotation of the entire grid (degrees).
        keep_x_sym: (unused, reserved).
        h_ax: optional axes handle.
        line_width: hexagon outline thickness.
        background_color: axes background color.

    Returns:
        centers: (N,2) array of hexagon centers (x,y)
        h_ax: axes handle (if plotting)
    """

    if not check_inputs(nx, ny, min_diam, n, align_to_origin, face_color,
                        edge_color, plotting_gap, crop_circ, do_plot,
                        rotate_deg, keep_x_sym, background_color):
        raise ValueError("Invalid arguments to create_hex_grid.")

    coord_x, coord_y = make_grid(
        nx, ny, min_diam, n, crop_circ, rotate_deg, align_to_origin
    )

    if do_plot:
        h_ax = plot_single_lattice(
            coord_x, coord_y,
            face_color, edge_color,
            min_diam, plotting_gap, rotate_deg,
            h_ax, background_color, line_width
        )

    centers = np.hstack([coord_x, coord_y])
    return centers, h_ax


# ============================================================
#                   Validation
# ============================================================

def check_inputs(nx, ny, min_diam, n, align_to_origin,
                 face_color, edge_color, plotting_gap, crop_circ,
                 do_plot, rotate_deg, keep_x_sym, background_color) -> bool:

    ok = True

    # --- numeric checks
    if any(not isinstance(v, (int, float)) for v in (nx, ny, n)) or nx < 0 or ny < 0:
        print("Error: nx, ny, n must be non-negative integers/floats.")
        ok = False

    if not isinstance(min_diam, (float, int)) or min_diam <= 0:
        print("Error: min_diam must be > 0.")
        ok = False

    if not isinstance(crop_circ, (float, int)) or crop_circ < 0:
        print("Error: crop_circ must be >= 0.")
        ok = False

    if not isinstance(plotting_gap, float) or not (0 <= plotting_gap < 1):
        print("Error: plotting_gap must be float in [0, 1).")
        ok = False

    if not isinstance(rotate_deg, (float, int)):
        print("Error: rotate_deg must be float/int.")
        ok = False

    # --- color validation
    VALID = {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}

    def valid_color(c):
        if c is None:
            return True
        if isinstance(c, str):
            return c in VALID
        if isinstance(c, list) and len(c) in (3, 4):
            return all(0 <= x <= 1 for x in c)
        return False

    if not valid_color(face_color) or not valid_color(edge_color) or not valid_color(background_color):
        print("Error: Invalid color format.")
        ok = False

    return ok


# ============================================================
#                   Plotting
# ============================================================

def plot_single_lattice(coord_x, coord_y,
                        face_color, edge_color,
                        min_diam, plotting_gap, rotate_deg,
                        h_ax=None, background_color=None,
                        line_width=0.2):

    if face_color is None:
        face_color = (1, 1, 1, 0)
    if edge_color is None:
        edge_color = 'k'

    if h_ax is None:
        fig = plt.figure(figsize=(5, 5))
        h_ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

    if background_color is not None:
        h_ax.set_facecolor(background_color)

    # RegularPolygon uses "radius" = distance from center to vertex, = side length
    radius = min_diam / np.sqrt(3) * (1 - plotting_gap)
    orientation = np.deg2rad(-rotate_deg)

    patches = []
    for x, y in zip(coord_x, coord_y):
        poly = mpatches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=radius,
            orientation=orientation
        )
        patches.append(poly)

    collection = PatchCollection(
        patches,
        edgecolor=edge_color,
        facecolor=face_color,
        linewidths=line_width
    )
    h_ax.add_collection(collection)

    h_ax.set_aspect("equal")
    h_ax.axis([
        coord_x.min() - 2 * min_diam,
        coord_x.max() + 2 * min_diam,
        coord_y.min() - 2 * min_diam,
        coord_y.max() + 2 * min_diam
    ])

    return h_ax


# ============================================================
#                   Grid generation
# ============================================================

def make_grid(nx, ny, min_diam, n, crop_circ, rotate_deg, align_to_origin):
    """
    Generate hex grid centers (flat-top layout).

    The pattern is:
        - odd rows shifted by +0.5
        - y spacing = sqrt(3)/2
    """

    ratio = np.sqrt(3) / 2

    # override nx, ny if n is provided
    if n > 0:
        ny = int(np.sqrt(n / ratio))
        nx = n // ny

    # create basic grid indices
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')

    # odd rows shifted by 0.5
    gx = gx.astype(float)
    gx[1::2, :] += 0.5

    gy = gy.astype(float) * ratio

    # flatten
    gx = gx.reshape(-1, 1) * min_diam
    gy = gy.reshape(-1, 1) * min_diam

    # center for rotation/cropping
    mid_x = (np.ceil(nx / 2) - 1 + 0.5 * (np.ceil(ny / 2) % 2 == 0)) * min_diam
    mid_y = (np.ceil(ny / 2) - 1) * ratio * min_diam

    # circular crop
    if crop_circ > 0:
        rad = np.sqrt((gx - mid_x) ** 2 + (gy - mid_y) ** 2)
        keep = rad.flatten() <= crop_circ
        gx, gy = gx[keep], gy[keep]

    # apply rotation
    if not np.isclose(rotate_deg, 0):
        R = np.array([
            [np.cos(np.deg2rad(rotate_deg)), np.sin(np.deg2rad(rotate_deg))],
            [-np.sin(np.deg2rad(rotate_deg)), np.cos(np.deg2rad(rotate_deg))]
        ])
        xy = np.hstack([gx - mid_x, gy - mid_y]) @ R.T
        gx, gy = np.hsplit(xy + [mid_x, mid_y], 2)

    # align to origin
    if align_to_origin:
        gx -= mid_x
        gy -= mid_y

    return gx, gy


# ============================================================
#                   Example
# ============================================================

def main():
    plt.ion()
    centers, ax = create_hex_grid(nx=5, ny=5, do_plot=True)
    plt.show(block=True)


if __name__ == "__main__":
    main()
