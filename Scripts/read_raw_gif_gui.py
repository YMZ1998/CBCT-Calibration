import sys
import numpy as np
import os
import imageio
import json
from tqdm import tqdm
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QFileDialog
)
from PyQt5.QtGui import QMovie

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.last_directory = self.load_last_directory()  # 加载上一次选择的目录
        self.gif_path = ""  # 用于保存生成的 GIF 路径
        self.initUI()

    def initUI(self):
        self.setWindowTitle('RAW to GIF Converter')
        self.setGeometry(200, 200, 300, 200)

        # 窗口居中
        screen = QApplication.primaryScreen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

        # 创建布局和控件
        layout = QVBoxLayout()

        self.info_label = QLabel("选择 RAW 文件目录并生成 GIF", self)
        layout.addWidget(self.info_label)

        self.select_button = QPushButton("选择目录", self)
        self.select_button.clicked.connect(self.select_directory)
        layout.addWidget(self.select_button)

        self.generate_button = QPushButton("生成 GIF", self)
        self.generate_button.clicked.connect(self.generate_gif)
        layout.addWidget(self.generate_button)

        self.replay_button = QPushButton("重播 GIF", self)  # 新增重播按钮
        self.replay_button.clicked.connect(self.replay_gif)
        self.replay_button.setEnabled(False)  # 初始禁用
        layout.addWidget(self.replay_button)

        self.gif_label = QLabel(self)
        layout.addWidget(self.gif_label)

        self.setLayout(layout)

    def load_last_directory(self):
        """从 JSON 文件加载上次选择的目录"""
        if os.path.exists("last_directory.json"):
            with open("last_directory.json", "r") as file:
                data = json.load(file)
                return data.get("last_directory", "")
        return ""

    def save_last_directory(self, directory):
        """保存上次选择的目录到 JSON 文件"""
        with open("last_directory.json", "w") as file:
            json.dump({"last_directory": directory}, file)

    def select_directory(self):
        # 打开上一次选择的目录
        self.directory = QFileDialog.getExistingDirectory(self, "选择 RAW 文件目录", self.last_directory)
        if self.directory:
            self.info_label.setText(f"选择的目录: {self.directory}")
            self.last_directory = self.directory  # 更新上一次选择的目录
            self.save_last_directory(self.last_directory)  # 保存新选择的目录

    def read_raw_image(self, file_path, width, height, dtype):
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        image = np.frombuffer(raw_data, dtype=dtype)
        image = image.reshape((height, width))
        return image

    def normalize_image(self, image):
        image = image.astype(np.float32)
        image_min = np.min(image)
        image_max = np.max(image)
        normalized_image = (image - image_min) / (image_max - image_min)
        return (normalized_image * 255).astype(np.uint8)

    def generate_gif(self):
        width = 512
        height = 512
        dtype = np.dtype('<i2')
        num_images = 180
        file_prefix = 'output_'

        images = []
        for i in tqdm(range(0, num_images, 3)):
            file_path = os.path.join(self.directory, f"{file_prefix}{i}.raw")
            if not os.path.exists(file_path):
                self.info_label.setText(f"未找到文件: {file_path}")
                return

            image_data = self.read_raw_image(file_path, width, height, dtype)
            normalized_image = self.normalize_image(image_data)
            images.append(normalized_image)

        self.gif_path = os.path.join(self.directory, 'output.gif')  # 保存 GIF 的路径
        imageio.mimsave(self.gif_path, images, duration=0.1)

        self.info_label.setText(f"GIF 动画已保存到: {self.gif_path}")
        self.display_gif(self.gif_path)  # 显示生成的 GIF
        self.replay_button.setEnabled(True)  # 启用重播按钮

    def display_gif(self, gif_path):
        movie = QMovie(gif_path)
        self.gif_label.setMovie(movie)
        movie.start()  # 启动 GIF 播放

    def replay_gif(self):
        """重新播放 GIF"""
        if self.gif_path:
            self.display_gif(self.gif_path)  # 重新显示 GIF


if __name__ == '__main__':
    app = QApplication(sys.argv)
    processor = ImageProcessor()
    processor.show()
    sys.exit(app.exec_())
