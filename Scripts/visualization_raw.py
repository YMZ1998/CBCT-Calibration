import os
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QSlider, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# 设置图像的宽度、高度和深度（通道数）
width = 512  # 图像宽度
height = 512  # 图像高度
depth = 512  # 深度（例如，图像切片的数量）

def read_raw_3d_image(file_path, width, height, depth):
    # 读取 RAW 三维图像
    with open(file_path, 'rb') as f:
        raw_data = f.read()

    # 将二进制数据转换为 NumPy 数组
    image = np.frombuffer(raw_data, dtype=np.dtype('<i2'))
    image = image.reshape((depth, height, width))  # 根据需要调整顺序

    return image

class ImageViewer(QMainWindow):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.initUI()

    def initUI(self):
        self.setWindowTitle("3D Image Viewer")
        self.setGeometry(100, 100, 1200, 800)  # 设置窗口初始大小

        # 创建一个主布局
        main_layout = QGridLayout()

        # 主视图（XY Plane）
        self.label_xy = QLabel("XY Plane")
        self.label_xy.setFixedSize(800, 600)  # 设置固定大小
        self.label_xy.setScaledContents(True)  # 允许按比例缩放
        main_layout.addWidget(self.label_xy, 0, 0, 1, 1)

        # 创建滑块
        self.slider_xy = QSlider(Qt.Horizontal)
        self.slider_xy.setRange(0, depth - 1)
        self.slider_xy.setValue(depth // 2)
        self.slider_xy.valueChanged.connect(self.update_xy_slice)
        main_layout.addWidget(QLabel("Select XY Plane Slice"), 2, 0)
        main_layout.addWidget(self.slider_xy, 3, 0)

        # 右侧布局（XZ 和 YZ Plane）
        self.label_xz = QLabel("XZ Plane")
        self.label_yz = QLabel("YZ Plane")

        main_layout.addWidget(self.label_xz, 0, 1)
        main_layout.addWidget(self.label_yz, 1, 1)

        # XZ Plane 滑块
        self.slider_xz = QSlider(Qt.Horizontal)
        self.slider_xz.setRange(0, width - 1)
        self.slider_xz.setValue(width // 2)
        self.slider_xz.valueChanged.connect(self.update_xz_slice)
        main_layout.addWidget(QLabel("Select XZ Plane Slice"), 2, 1)
        main_layout.addWidget(self.slider_xz, 3, 1)

        # YZ Plane 滑块
        self.slider_yz = QSlider(Qt.Horizontal)
        self.slider_yz.setRange(0, height - 1)
        self.slider_yz.setValue(height // 2)
        self.slider_yz.valueChanged.connect(self.update_yz_slice)
        main_layout.addWidget(QLabel("Select YZ Plane Slice"), 4, 1)
        main_layout.addWidget(self.slider_yz, 5, 1)

        # 创建一个 QWidget 作为中央窗口
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 显示初始截面
        self.show_slices()

    def show_slices(self):
        self.update_xy_slice(self.slider_xy.value())
        self.update_xz_slice(self.slider_xz.value())
        self.update_yz_slice(self.slider_yz.value())

    def update_xy_slice(self, value):
        xy_slice = self.image[value]
        pixmap = self.array_to_pixmap(xy_slice)
        self.label_xy.setPixmap(pixmap)
        self.label_xy.setFixedSize(pixmap.size())  # 设置 QLabel 大小为图像大小

    def update_xz_slice(self, value):
        xz_slice = self.image[:, value, :]
        pixmap = self.array_to_pixmap(xz_slice)
        self.label_xz.setPixmap(pixmap)

    def update_yz_slice(self, value):
        yz_slice = self.image[:, :, value]
        pixmap = self.array_to_pixmap(yz_slice)
        self.label_yz.setPixmap(pixmap)

    def array_to_pixmap(self, array):
        # 将 NumPy 数组转换为 QPixmap
        # Normalize the array to the range [0, 255] (for 8-bit images)
        normalized_array = np.interp(array, (array.min(), array.max()), (0, 255)).astype(np.uint8)

        # Create a QImage from the NumPy array
        height, width = normalized_array.shape
        qimage = QImage(normalized_array.data, width, height, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap
        return QPixmap.fromImage(qimage)


if __name__ == "__main__":
    # 修改为你自己的 RAW 文件路径
    file_path = r"D:\Data\result\result.raw"

    # 读取三维图像
    image = read_raw_3d_image(file_path, width, height, depth)

    app = QApplication(sys.argv)
    viewer = ImageViewer(image)
    viewer.show()
    sys.exit(app.exec_())
