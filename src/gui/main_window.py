import sys
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt6.QtGui import QPixmap, QPainter, QColor, QMouseEvent, QCursor
from PyQt6.QtCore import Qt, QPoint, QTimer

from src.gui.chat_dialog import ChatDialog # 确保 chat_dialog.py 在 src/gui/ 目录下

class PetWindow(QWidget):
    def __init__(self, image_path="src/assets/pet.png"):
        super().__init__()
        self.image_path = image_path
        self.chat_dialog = None
        self.offset = QPoint()

        self.init_ui()
        self.load_pet_image()

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |          # 无边框
            Qt.WindowType.WindowStaysOnTopHint |         # 窗口总在最前
            Qt.WindowType.Tool                           # 不在任务栏显示图标（在某些系统上）
        )
        # 设置背景透明的关键属性
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self.label = QLabel(self)
        # 如果需要固定大小，可以取消注释下面这行，并调整为你图片的大小
        # self.setFixedSize(128, 128) # 假设图片是128x128

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setContentsMargins(0,0,0,0) # 确保布局没有额外的边距
        self.setLayout(layout)

        self.show()

    def load_pet_image(self):
        pixmap = QPixmap(self.image_path)
        if pixmap.isNull():
            print(f"错误：无法加载图片 {self.image_path}")
            # 创建一个红色方块作为错误提示
            error_pixmap = QPixmap(100, 100)
            error_pixmap.fill(QColor("red"))
            painter = QPainter(error_pixmap)
            painter.setPen(QColor("white"))
            painter.drawText(error_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "图片加载失败")
            painter.end()
            self.label.setPixmap(error_pixmap)
            self.resize(error_pixmap.size())
        else:
            self.label.setPixmap(pixmap)
            self.resize(pixmap.size()) # 根据图片大小调整窗口大小

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.offset = event.globalPosition().toPoint() - self.pos()
            self.show_chat_dialog() # 左键点击也打开对话框
        elif event.button() == Qt.MouseButton.RightButton:
            # 可以在这里添加右键菜单，例如退出应用
            # QApplication.instance().quit() # 简单示例：右键退出
            pass

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.offset)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # 双击可以考虑做其他事情，或者也打开对话框
        # self.show_chat_dialog()
        pass

    def show_chat_dialog(self):
        if self.chat_dialog is None or not self.chat_dialog.isVisible():
            self.chat_dialog = ChatDialog(self) # 将主窗口作为父窗口
            # 计算对话框位置，使其出现在宠物旁边
            pet_pos = self.pos()
            pet_width = self.width()
            dialog_width = self.chat_dialog.width()
            
            # 尝试放在宠物右边
            dialog_x = pet_pos.x() + pet_width
            dialog_y = pet_pos.y()

            # 检查是否会超出屏幕右边界
            screen_geometry = QApplication.primaryScreen().availableGeometry()
            if dialog_x + dialog_width > screen_geometry.right():
                dialog_x = pet_pos.x() - dialog_width # 放到左边

            # 检查是否会超出屏幕上边界
            if dialog_y < screen_geometry.top():
                dialog_y = screen_geometry.top()
            # 检查是否会超出屏幕下边界
            if dialog_y + self.chat_dialog.height() > screen_geometry.bottom():
                dialog_y = screen_geometry.bottom() - self.chat_dialog.height()

            self.chat_dialog.move(dialog_x, dialog_y)
            self.chat_dialog.show()
        else:
            self.chat_dialog.activateWindow() # 如果已存在，则激活
            self.chat_dialog.raise_()         # 并提升到顶层

if __name__ == '__main__':
    # 这部分代码仅用于单独测试 PetWindow
    app = QApplication(sys.argv)
    # 请确保在 src/assets/ 目录下有一个 pet.png 文件
    # 或者提供一个有效的图片路径
    # 例如: pet_window = PetWindow("path/to/your/image.png")
    pet_window = PetWindow() # 使用默认路径 "src/assets/pet.png"
    if pet_window.label.pixmap().isNull():
        print("主窗口：宠物图片未加载，请检查路径或文件。")
    sys.exit(app.exec())