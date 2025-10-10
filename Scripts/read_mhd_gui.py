import sys
import os
import json
import SimpleITK as sitk
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class MHDViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # 读取上次打开的目录
        self.last_directory = self.load_last_directory()

        # 初始化 UI
        self.init_ui()

    def load_last_directory(self):
        """从 JSON 文件加载上次打开的目录"""
        try:
            with open("last_directory.json", "r") as f:
                data = json.load(f)
                return data.get("last_directory", os.getcwd())  # 默认当前工作目录
        except FileNotFoundError:
            return os.getcwd()  # 如果文件不存在，返回当前工作目录

    def save_last_directory(self, directory):
        """将当前目录保存到 JSON 文件"""
        with open("last_directory.json", "w") as f:
            json.dump({"last_directory": directory}, f)

    def init_ui(self):
        # 创建一个 QVBoxLayout 来布局控件
        layout = QVBoxLayout()

        # 创建 QLabel 显示图像
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # 创建切片信息显示标签
        self.slice_info_label = QLabel(self)
        layout.addWidget(self.slice_info_label)

        # 创建 QSlider 用于切换切片
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)  # 初始最大值设置为 0，后面会更新
        self.slider.valueChanged.connect(self.update_slice)
        layout.addWidget(self.slider)

        # 创建按钮以选择 MHD 文件
        self.open_button = QPushButton("Open MHD File", self)
        self.open_button.clicked.connect(self.open_file)
        layout.addWidget(self.open_button)

        # 设置中心窗口部件
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 设置窗口属性
        self.setWindowTitle("MHD Viewer")
        self.setGeometry(100, 100, 520, 520)
        self.show()

    def open_file(self):
        # 打开文件对话框，选择 MHD 文件
        mhd_file_path, _ = QFileDialog.getOpenFileName(self, "Open MHD File", self.last_directory, "MHD Files (*.mhd)")

        if mhd_file_path:  # 确保用户选择了文件
            self.load_image(mhd_file_path)
            self.save_last_directory(os.path.dirname(mhd_file_path))  # 保存当前文件的上一级目录

    def load_image(self, mhd_file_path):
        # 读取 MHD 文件并转换为 NumPy 数组
        image = sitk.ReadImage(mhd_file_path)
        self.image_array = sitk.GetArrayFromImage(image)  # NumPy 数组，维度 (z, y, x)

        if len(self.image_array.shape) != 3:
            print("Error: The image is not a 3D image.")
            return
        # 获取图像的 z 维度大小（切片数量）和 y、x 尺寸
        self.num_slices = self.image_array.shape[0]
        self.height = self.image_array.shape[1]
        self.width = self.image_array.shape[2]

        # 整体数据归一化
        self.normalized_array = self.normalize_image_data(self.image_array)

        # 更新滑块的最大值
        self.slider.setMaximum(self.num_slices - 1)
        self.slider.setValue(self.num_slices // 2)  # 初始显示中间的切片

        # 显示初始切片
        self.update_slice(self.slider.value())

    def normalize_image_data(self, array):
        # 计算全局最小值和最大值
        min_val = np.min(array)
        max_val = np.max(array)

        # 归一化到 0-255 范围
        normalized_array = (array - min_val) / (max_val - min_val) * 255.0
        return normalized_array.astype(np.uint8)

    def update_slice(self, slice_index):
        # 根据 slider 的值更新显示的切片
        slice_image = self.normalized_array[slice_index, :, :]

        # 将 NumPy 数组转换为 QImage
        qimage = self.numpy_to_qimage(slice_image)

        # 显示 QImage
        self.image_label.setPixmap(QPixmap.fromImage(qimage))

        # 更新切片信息
        self.slice_info_label.setText(f"Slice: {slice_index + 1}/{self.num_slices} (Size: {self.width}x{self.height})")

    def numpy_to_qimage(self, array):
        # 将 NumPy 数组转换为 QImage
        height, width = array.shape
        qimage = QImage(array.data, width, height, QImage.Format_Grayscale8)
        return qimage


if __name__ == '__main__':
    app = QApplication(sys.argv)

    viewer = MHDViewer()
    sys.exit(app.exec_())
