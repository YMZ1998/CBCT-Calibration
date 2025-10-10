import os
import sys
import json
import numpy as np
import SimpleITK as sitk

from PyQt5.QtWidgets import (
    QApplication, QWidget, QListWidget, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage.metrics import mean_squared_error, structural_similarity as ssim


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)


CONFIG_PATH = "MHDCompareUI.json"


class ImageCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)
        self.axs = self.fig.subplots(2, 3)  # 初始创建一次

    def show_images(self, images, titles, title):
        # 清除整个画布
        self.fig.clf()
        self.axs = self.fig.subplots(2, 3)
        self.fig.suptitle(title)

        for ax, img, title in zip(self.axs.flat, images, titles):
            ax.imshow(img, cmap="gray" if "Image" in title else "jet")
            ax.set_title(title)
            ax.axis("off")
            if "Diff" in title:
                self.fig.colorbar(ax.images[0], ax=ax, shrink=0.7)

        self.draw()


class MHDCompareUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MHD 图像对比工具")
        self.resize(1200, 800)

        self.root_path = ""
        self.case_list = QListWidget()
        self.case_list.itemClicked.connect(self.load_case)

        self.btn_select_dir = QPushButton("选择根目录")
        self.btn_select_dir.clicked.connect(self.select_root_dir)

        self.canvas = ImageCanvas()

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.btn_select_dir)
        self.left_layout.addWidget(self.case_list)

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(self.left_layout)
        right_widget = QWidget()
        right_widget.setLayout(self.right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])

        layout = QHBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.load_last_path()

    def load_last_path(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                path = data.get("last_dir", "")
                if os.path.isdir(path):
                    self.root_path = path
                    self.update_case_list()

    def save_last_path(self):
        with open(CONFIG_PATH, "w") as f:
            json.dump({"last_dir": self.root_path}, f)

    def select_root_dir(self):
        start_dir = self.root_path if self.root_path and os.path.isdir(self.root_path) else os.path.expanduser("~")
        dir_path = QFileDialog.getExistingDirectory(self, "选择根目录", start_dir)
        if dir_path:
            self.root_path = dir_path
            self.save_last_path()
            self.update_case_list()

    def update_case_list(self):
        self.case_list.clear()
        if not os.path.isdir(self.root_path):
            return
        for name in sorted(os.listdir(self.root_path)):
            case_dir = os.path.join(self.root_path, name)
            if os.path.isdir(case_dir):
                self.case_list.addItem(name)

    def load_case(self, item):
        case_name = item.text()
        case_dir = os.path.join(self.root_path, case_name)
        paths = {
            "a": os.path.join(case_dir, "a_mhd.mhd"),
            "drra": os.path.join(case_dir, "drra.mhd"),
            "b": os.path.join(case_dir, "b_mhd.mhd"),
            "drrb": os.path.join(case_dir, "drrb.mhd"),
        }

        try:
            imgs = {k: sitk.GetArrayFromImage(sitk.ReadImage(p)) for k, p in paths.items()}
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取图像失败：\n{str(e)}")
            return

        if imgs["a"].shape != imgs["drra"].shape or imgs["b"].shape != imgs["drrb"].shape:
            QMessageBox.warning(self, "尺寸不匹配", "图像对的尺寸不一致，无法比较。")
            return

        norm = {k: normalize_image(v) for k, v in imgs.items()}
        diff1 = norm["a"] - norm["drra"]
        diff2 = norm["b"] - norm["drrb"]

        mse1 = mean_squared_error(norm["a"].flatten(), norm["drra"].flatten())
        mse2 = mean_squared_error(norm["b"].flatten(), norm["drrb"].flatten())
        ssim1 = ssim(norm["a"], norm["drra"], data_range=1.0)
        ssim2 = ssim(norm["b"], norm["drrb"], data_range=1.0)

        print(f"Case: {case_name}\n"
              f"Pair A  MSE: {mse1:.6f}    SSIM: {ssim1:.6f}\n"
              f"Pair B  MSE: {mse2:.6f}    SSIM: {ssim2:.6f}")

        self.canvas.show_images(
            images=[norm["a"], norm["drra"], diff1, norm["b"], norm["drrb"], diff2],
            titles=["Image 1 - A", "Image 2 - A", "Diff A",
                    "Image 1 - B", "Image 2 - B", "Diff B"],
            title=f"Case: {case_name}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MHDCompareUI()
    ui.show()
    sys.exit(app.exec_())
