from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt

class ChatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("聊天中...")
        self.setMinimumWidth(300)

        self.layout = QVBoxLayout(self)

        self.message_label = QLabel("你好，我是哈基米！我们开始聊天吧。", self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.message_label)

        self.ok_button = QPushButton("好的", self)
        self.ok_button.clicked.connect(self.accept) # accept() 会关闭对话框并返回 QDialog.DialogCode.Accepted
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def set_message(self, message):
        self.message_label.setText(message)

if __name__ == '__main__':
    # 这部分代码仅用于单独测试 ChatDialog
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dialog = ChatDialog()
    dialog.show()
    sys.exit(app.exec())