import sys
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import PetWindow

def main():
    app = QApplication(sys.argv)

    # 确保你的图片 pet.png 放在 src/assets/ 目录下
    # PetWindow 默认会加载 "src/assets/pet.png"
    pet = PetWindow()
    
    if pet.label.pixmap() is None or pet.label.pixmap().isNull():
        print("错误：未能加载宠物图片，程序将退出。")
        print(f"请确保图片 'pet.png' 存在于 'desktop_pet_project/src/assets/' 目录下。")
        # 在实际应用中，这里可以弹出一个错误对话框
        # 或者提供一个默认的占位符图像
        # return # 直接退出可能不是最佳用户体验，但对于初期调试是明确的

    sys.exit(app.exec())

if __name__ == '__main__':
    main()