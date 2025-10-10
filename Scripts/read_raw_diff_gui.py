import json
import os
import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

SETTINGS_FILE = "RawCompare.json"


def read_raw_image(filename, width, height, dtype=np.uint16):
    file_size = width * height * np.dtype(dtype).itemsize
    with open(filename, 'rb') as f:
        raw_data = f.read(file_size)
    image = np.frombuffer(raw_data, dtype=dtype).reshape((height, width)).astype(np.float32)
    return image


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6):
        self.fig = Figure(figsize=(width, height))
        super().__init__(self.fig)
        self.axs = self.fig.subplots(2, 3)  # axs.shape = (2, 3)
        self.fig.tight_layout()
        self.setParent(parent)

    def show_images(self, image1, image2, diff1, image3, image4, diff2, titles):
        images = [image1, image2, diff1, image3, image4, diff2]
        self.fig.clf()
        self.axs = self.fig.subplots(2, 3)
        self.fig.tight_layout()

        for ax, img, title in zip(self.axs.flat, images, titles):
            ax.clear()
            im = ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
            self.fig.colorbar(im, ax=ax, shrink=0.6)

        self.draw()


class RawCompare(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Compare")
        self.resize(1000, 800)
        self.setWindowIcon(QIcon())  # 可自定义图标路径

        self.width = 2130
        self.height = 2130
        self.file1 = ""
        self.file2 = ""
        self.root_path = ""

        # 字体
        btn_font = QFont("Arial", 11, QFont.Bold)
        label_font = QFont("Arial", 10, QFont.Bold)

        self.label_root = QLabel("当前目录：未选择")
        self.label_root.setFont(label_font)

        self.btn_set_dir = QPushButton("选择目录")
        self.btn_set_dir.setFont(btn_font)
        self.btn_set_dir.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 6px 12px; border-radius: 4px;")
        self.btn_set_dir.clicked.connect(self.set_root_directory_dialog)

        self.raw_list = QListWidget()
        self.raw_list.itemChanged.connect(self.on_item_changed)
        self.raw_list.setStyleSheet("""
            QListWidget::item {
                padding-left: 4px;
                font-size: 10pt;
            }
            QListWidget::item:selected {
                background-color: #87CEFA;
            }
            QListWidget::indicator {
                width: 14px;
                height: 14px;
                margin-left: 2px;
                margin-right: 5px;
            }
        """)
        self.raw_list.itemDoubleClicked.connect(self.on_item_double_clicked)

        self.btn_compare = QPushButton("比较图像")
        self.btn_compare.setFont(btn_font)
        self.btn_compare.setStyleSheet("""
            background-color: #2196F3;
            color: white;
            padding: 14px 30px;
            border-radius: 5px;
        """)
        self.btn_compare.setEnabled(False)
        self.btn_compare.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.btn_compare.clicked.connect(self.compare_images)

        self.canvas = ImageCanvas(self)

        # 左侧布局（目录+列表）
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label_root)
        left_layout.addWidget(self.btn_set_dir)
        left_layout.addWidget(self.raw_list)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        # 主垂直布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.btn_compare, 0, Qt.AlignHCenter)
        main_layout.addWidget(self.canvas)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        self.setLayout(main_layout)
        self.load_settings()

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                last_dir = data.get("last_dir", "")
                if last_dir and os.path.isdir(last_dir):
                    self.set_root_directory(last_dir)
            except Exception as e:
                print(f"加载配置失败: {e}")

    def save_settings(self):
        data = {
            "last_dir": self.root_path if self.root_path else ""
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def set_root_directory_dialog(self):
        start_dir = self.root_path if self.root_path and os.path.isdir(self.root_path) else os.path.expanduser("~")
        dir_path = QFileDialog.getExistingDirectory(self, "选择根目录", start_dir)
        if dir_path:
            self.set_root_directory(dir_path)

    def set_root_directory(self, dir_path):
        self.root_path = dir_path
        self.label_root.setText(f"当前目录：{dir_path}")
        self.raw_list.clear()
        raw_files = self.find_all_raw_files(dir_path)
        for f in raw_files:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.raw_list.addItem(item)
        self.save_settings()

        self.file1 = ""
        self.file2 = ""
        self.btn_compare.setEnabled(False)

    def find_all_raw_files(self, root):
        raw_files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith('A.raw'):
                    raw_files.append(os.path.join(dirpath, f))
        return sorted(raw_files)

    def on_item_changed(self, item):
        checked_items = [self.raw_list.item(i) for i in range(self.raw_list.count())
                         if self.raw_list.item(i).checkState() == Qt.Checked]

        if len(checked_items) > 2:
            item.setCheckState(Qt.Unchecked)
            print("最多只能选择两个文件作为对比图像")
            return

        if len(checked_items) == 0:
            self.file1 = ""
            self.file2 = ""
        elif len(checked_items) == 1:
            self.file1 = checked_items[0].text()
            self.file2 = ""
        else:
            self.file1 = checked_items[0].text()
            self.file2 = checked_items[1].text()

        self.btn_compare.setEnabled(bool(self.file1 and self.file2))

    def on_item_double_clicked(self, item):
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)

    def compare_images(self):
        image1 = read_raw_image(self.file1, self.width, self.height)
        image2 = read_raw_image(self.file2, self.width, self.height)
        image3 = read_raw_image(self.file1.replace('A.raw', 'B.raw'), self.width, self.height)
        image4 = read_raw_image(self.file2.replace('A.raw', 'B.raw'), self.width, self.height)
        diff1 = image1 - image2
        diff2 = image3 - image4
        self.canvas.show_images(
            image1, image2, diff1, image3, image4, diff2,
            [os.path.basename(os.path.dirname(self.file1)), os.path.basename(os.path.dirname(self.file2)),
             "A Difference", os.path.basename(os.path.dirname(self.file1)),
             os.path.basename(os.path.dirname(self.file2)), "B Difference"]
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RawCompare()
    window.show()
    sys.exit(app.exec_())
