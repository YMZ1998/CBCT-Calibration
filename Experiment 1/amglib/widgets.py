import os
import numpy as np
from ipyfilechooser import FileChooser
import ipywidgets as widgets
import sys

from rich.jupyter import display

import amglib.readers as rd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


class FileSelector:
    FILE_PATTERNS = ['*.png', '*.jpg', '*.npy', '*.fits', '*.tif', '*.tiff']

    @property
    def file_name(self):
        return self._fileinfo["name"]

    @property
    def file_path(self):
        return self._fileinfo["path"]

    @property
    def file_mask(self):
        return self._fileinfo["mask"]

    @property
    def first_index(self):
        return self._fileinfo["first"]

    @property
    def last_index(self):
        return self._fileinfo["last"]

    @property
    def info(self):
        return self._fileinfo

    def __init__(self, start_dir='.'):
        self.file_chooser = FileChooser(
            os.path.realpath(start_dir),
            show_only_dirs=False,
            select_desc="Browse file",
            change_desc="Change file",
            #             filter_pattern=";".join(self.FILE_PATTERNS)
        )
        self.file_chooser.layout = widgets.Layout(width='800px')
        self.file_chooser._select.layout = widgets.Layout(width="200px")
        self.file_chooser.register_callback(self.on_file_selected)

    def on_file_selected(self, chooser):
        selected = chooser.selected
        if selected:
            self.analyze_file(selected)

    def analyze_file(self, filename):
        # Implement your analysis logic here
        selected_file = self.file_chooser.selected
        path = os.path.dirname(selected_file)
        fname = os.path.basename(selected_file)

        ext = selected_file.split('.')[-1]
        flist = rd.list_matching_files(path, r'_'.join(fname.split('_')[:-1]) + '*.' + ext)
        fmask = rd.make_file_mask(selected_file)
        first, last = rd.find_first_last_indices(flist)

        self._fileinfo = {"name": selected_file,
                          "mask": fmask,
                          "path": path,
                          "ext": ext,
                          "first": first,
                          "last": last,
                          "files": flist}

    def display(self):
        display(self.file_chooser)


class ROISelector:
    def __init__(self, image: np.ndarray, cmap='viridis', title=None, show_colorbar=False, quantile=None):
        self._image = image
        self._roi_coords = None

        self._fig, self._ax = plt.subplots()
        if quantile is not None:
            vmin = np.quantile(self._image, (1 - quantile) / 2)
            vmax = np.quantile(self._image, 0.5 + quantile / 2)
        else:
            vmin = self._image.min()
            vmax = self._image.max()

        a = self._ax.imshow(self._image, cmap=cmap, vmin=vmin, vmax=vmax)

        if show_colorbar:
            self._fig.colorbar(a)

        if title is not None:
            self._ax.set_title(title)

        self._selector = RectangleSelector(
            self._ax,
            self._on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=2,
            minspany=2,
            spancoords='pixels',
            interactive=True
        )
        self._fig.canvas.header_visible = False
        self._fig.canvas.footer_visible = False
        self._fig.canvas.toolbar_visible = False
        plt.show()

    def _on_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self._roi_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        print("ROI selected:", self._roi_coords)

    def __del__(self):
        # Safely close the figure when object is deleted
        if hasattr(self, "_fig"):
            plt.close(self._fig)
            # Optional: print for debugging
            print("ROISelector figure closed.")

    @property
    def roi(self):
        """Return the selected ROI as (x1, y1, x2, y2) or None if not yet selected."""
        return self._roi_coords

    @property
    def roi_image(self):
        """Return the extracted ROI image (as a NumPy array) or None if not yet selected."""
        if self._roi_coords is None:
            return None
        x1, y1, x2, y2 = self._roi_coords
        return self._image[y1:y2, x1:x2]
