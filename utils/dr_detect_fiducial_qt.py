import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from SymmetryEstimation.utils import read_raw_image
from utils.correct_dr import normalize_and_correct_dr_image


class FiducialDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DR Fiducial Detection")
        self.image = None
        self.circles = None
        self.image_size = 2130

        # === UI组件 ===
        self.label = QLabel("请加载图像")
        self.label.setAlignment(Qt.AlignCenter)

        # 窗位
        self.slider_level = QSlider(Qt.Horizontal)
        self.slider_level.setRange(0, 4000)
        self.slider_level.setValue(1000)
        self.slider_level.setTickInterval(100)
        self.spin_level = QSpinBox()
        self.spin_level.setRange(0, 4000)
        self.spin_level.setValue(1000)

        # 窗宽
        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 4000)
        self.slider_width.setValue(1000)
        self.slider_width.setTickInterval(100)
        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 4000)
        self.spin_width.setValue(1000)

        self.load_btn = QPushButton("加载图像")
        self.detect_btn = QPushButton("检测金标")

        # === 布局 ===
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("窗位"))
        level_layout.addWidget(self.slider_level)
        level_layout.addWidget(self.spin_level)

        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("窗宽"))
        width_layout.addWidget(self.slider_width)
        width_layout.addWidget(self.spin_width)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.detect_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addLayout(level_layout)
        main_layout.addLayout(width_layout)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

        # === 信号绑定 ===
        self.load_btn.clicked.connect(self.load_image)
        self.detect_btn.clicked.connect(self.detect_fiducial)

        # 滑动条与数字框联动
        self.slider_level.valueChanged.connect(lambda val: self.spin_level.setValue(val))
        self.slider_width.valueChanged.connect(lambda val: self.spin_width.setValue(val))
        self.spin_level.valueChanged.connect(lambda val: self.slider_level.setValue(val))
        self.spin_width.valueChanged.connect(lambda val: self.slider_width.setValue(val))

        # 滑动条或数字改变时更新显示
        self.slider_level.valueChanged.connect(self.update_display)
        self.slider_width.valueChanged.connect(self.update_display)
        self.spin_level.valueChanged.connect(self.update_display)
        self.spin_width.valueChanged.connect(self.update_display)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择RAW图像", r"D:\Data\cbct\DR0707\body\1", "RAW Files (*.raw)")
        if fname:
            image = read_raw_image(fname, self.image_size, self.image_size, dtype=np.uint16)
            self.image = normalize_and_correct_dr_image(image)
            self.update_display()

    def window_level_transform(self, img, level, width):
        """根据窗宽窗位调整灰度"""
        # img = np.clip(img, level - width//2, level + width//2)
        # img = ((img - (level - width//2)) / width * 255).astype(np.uint8)
        min_val, max_val = np.min(img), np.max(img)
        img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return img

    def detect_fiducial(self):
        if self.image is None:
            return
        level = self.slider_level.value()
        width = self.slider_width.value()
        norm = self.window_level_transform(self.image, level, width)
        blurred = cv2.GaussianBlur(norm, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=25
        )
        self.circles = circles
        self.update_display()

    def update_display(self):
        if self.image is None:
            return
        level = self.slider_level.value()
        width = self.slider_width.value()
        norm = self.window_level_transform(self.image, level, width)
        output = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

        if self.circles is not None:
            for (x, y, r) in np.uint16(np.around(self.circles[0, :])):
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), -1)

        h, w, ch = output.shape
        bytes_per_line = ch * w
        qimg = QImage(output.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(800, 800, Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FiducialDetector()
    win.show()
    sys.exit(app.exec_())
