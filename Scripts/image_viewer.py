
import json
import os
import sys

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QColor, QPalette, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QVBoxLayout, QWidget, QFileDialog,
    QAction, QLabel, QFrame, QShortcut
)

CONFIG_FILE = 'ImageViewer.json'


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self._zoom = 0
        self._empty = True

        self._scene = QGraphicsScene(self)
        self._image_item = QGraphicsPixmapItem()
        self._scene.addItem(self._image_item)
        self.setScene(self._scene)

        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor(180, 180, 180, 80))
        self.setFrameShape(QFrame.NoFrame)

    def has_image(self):
        return not self._empty

    def set_image(self, pixmap):
        self._zoom = 0
        self._image_item.setPixmap(pixmap)
        self._empty = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_in_view()

    def fit_in_view(self):
        rect = QRectF(self._image_item.pixmap().rect())
        if not rect.isNull():
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
            self.scale(factor, factor)
            self._zoom = 0

    def wheelEvent(self, event):
        if self.has_image():
            factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
            self.scale(factor, factor)
            self._zoom += event.angleDelta().y() / 120

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.has_image():
            mouse_pos = event.pos()
            scene_pos = self.mapToScene(mouse_pos)
            image_pos = self._image_item.mapFromScene(scene_pos)

            if self._image_item.pixmap().rect().contains(image_pos.toPoint()):
                color = QColor(self._image_item.pixmap().toImage().pixel(image_pos.toPoint()))
                self.parent.update_pixel_label(
                    f'Pixel Value: R={color.red()} G={color.green()} B={color.blue()} '
                    f'(X={image_pos.x():.0f}, Y={image_pos.y():.0f})',
                    color
                )
            else:
                self.parent.update_pixel_label('Pixel Value: Out of bounds', None)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.last_open_path = ''
        self.load_config()
        self.initUI()
        self.center()
        self.init_shortcuts()

    def initUI(self):
        self.setWindowTitle('Image Viewer')

        self.graphicsView = GraphicsView(self)
        self.pixelLabel = QLabel('Pixel Value:', self)
        self.pixelLabel.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.graphicsView)
        layout.addWidget(self.pixelLabel)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.create_actions()
        self.create_menus()

        self.setGeometry(100, 100, 800, 600)

    def center(self):
        screen = QApplication.primaryScreen()
        screen_size = screen.availableSize()
        self.move((screen_size.width() - self.width()) // 2, (screen_size.height() - self.height()) // 2)

    def create_actions(self):
        self.openFile = QAction('Open', self)
        self.openFile.setShortcut('Ctrl+O')
        self.openFile.setStatusTip('Open new Image')
        self.openFile.triggered.connect(self.show_file_dialog)

        self.zoomIn = QAction('Zoom In', self)
        self.zoomIn.setShortcut('Ctrl+=')
        self.zoomIn.setStatusTip('Zoom In')
        self.zoomIn.triggered.connect(lambda: self.graphicsView.scale(1.25, 1.25))

        self.zoomOut = QAction('Zoom Out', self)
        self.zoomOut.setShortcut('Ctrl+-')
        self.zoomOut.setStatusTip('Zoom Out')
        self.zoomOut.triggered.connect(lambda: self.graphicsView.scale(0.8, 0.8))

        self.resetZoom = QAction('Reset Zoom', self)
        self.resetZoom.setShortcut('Ctrl+R')
        self.resetZoom.setStatusTip('Reset Zoom')
        self.resetZoom.triggered.connect(self.graphicsView.fit_in_view)

    def create_menus(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.openFile)

        viewMenu = menubar.addMenu('&View')
        viewMenu.addAction(self.zoomIn)
        viewMenu.addAction(self.zoomOut)
        viewMenu.addAction(self.resetZoom)

    def init_shortcuts(self):
        self.shortcut_quit = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.shortcut_quit.activated.connect(self.close)

    def show_file_dialog(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", self.last_open_path,
            "Images (*.png *.xpm *.jpg *.bmp *.tif);;All Files (*)",
            options=options
        )
        if filePath:
            self.last_open_path = os.path.dirname(filePath)
            self.save_config()
            self.load_image(filePath)

    def load_image(self, file_path):
        self.graphicsView.set_image(QPixmap(file_path))

    def update_pixel_label(self, text, color):
        self.pixelLabel.setText(text)
        if color:
            textColor = "white" if color == QColor(0, 0, 0) else "black"
            self.pixelLabel.setStyleSheet(f"QLabel {{ color: {textColor}; }}")

            palette = self.pixelLabel.palette()
            palette.setColor(QPalette.Window, color)
            self.pixelLabel.setAutoFillBackground(True)
            self.pixelLabel.setPalette(palette)
            self.graphicsView.setBackgroundBrush(color)
        else:
            self.pixelLabel.setAutoFillBackground(False)
            self.pixelLabel.setStyleSheet("")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                config = json.load(file)
                self.last_open_path = config.get('last_open_path', '')

    def save_config(self):
        config = {'last_open_path': self.last_open_path}
        with open(CONFIG_FILE, 'w') as file:
            json.dump(config, file)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
