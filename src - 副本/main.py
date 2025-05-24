import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from PyQt6.QtWidgets import QApplication, QMessageBox
from src.utils.config_manager import ConfigManager
from src.llm.gemini_client import GeminiClient
from src.gui.main_window import PetWindow
from src.gui.chat_dialog import ChatDialog
from src.database.mongo_handler import MongoHandler
from PIL import Image, ImageDraw, ImageFont

config_manager_global = None
gemini_client_global = None
pet_window_global = None
chat_dialog_global = None
assets_path_global = None
mongo_handler_global = None
avatar_base_path_global = None
pet_avatar_path_global = None
user_avatar_path_global = None


def create_placeholder_avatar(
    image_path: str, text: str, size=(64, 64), bg_color=(128, 128, 128, 200)
):
    try:
        img = Image.new("RGBA", size, bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font_size = size[1] // 2
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = (size[0] - text_width) / 2
        y = (size[1] - text_height) / 2
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
        img.save(image_path)
        print(f"已创建占位头像图片: {image_path}")
    except ImportError:
        print(
            f"Pillow 未安装，无法为 {image_path} 创建占位头像图片。请手动放置一张图片。"
        )
    except Exception as e:
        print(f"为 {image_path} 创建占位头像图片失败: {e}")


def setup_environment_and_config():
    global config_manager_global, assets_path_global, project_root
    global avatar_base_path_global, pet_avatar_path_global, user_avatar_path_global
    os.makedirs(os.path.join(project_root, "config"), exist_ok=True)
    assets_path_global = os.path.normpath(os.path.join(project_root, "src", "assets"))
    os.makedirs(assets_path_global, exist_ok=True)
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    config_file_relative_path = os.path.join("config", "settings.ini")
    actual_config_file_path = os.path.join(project_root, config_file_relative_path)
    if not os.path.exists(actual_config_file_path):
        print(f"警告：配置文件 {actual_config_file_path} 不存在。将创建一个模板。")
        try:
            with open(actual_config_file_path, "w", encoding="utf-8") as cf:
                cf.write(
                    "[GEMINI]\nAPI_KEY = YOUR_API_KEY_HERE\nMODEL_NAME = gemini-1.5-flash-latest\nHTTP_PROXY =\nHTTPS_PROXY =\n\n"
                )
                cf.write(
                    "[PET]\nINITIAL_IMAGE_FILENAME = default.png\nNAME = 小助手\nPERSONA = 你是一个友好、乐于助人的桌面宠物，名叫“小助手”。你会用亲切的语气和用户交流。\n\n"
                )
                cf.write("[USER]\nNAME = 主人\n\n")
                cf.write(
                    "[AVATARS]\nAVATAR_BASE_PATH = src/assets/icon\nPET_AVATAR_FILENAME = bot.png\nUSER_AVATAR_FILENAME = user.png\n\n"
                )
                cf.write(
                    "[MONGODB]\nCONNECTION_STRING = mongodb://localhost:27017/\nDATABASE_NAME = desktop_pet_db\nCOLLECTION_NAME = chat_history\nHISTORY_COUNT_FOR_PROMPT = 5\n"
                )
            QMessageBox.information(
                None,
                "配置文件创建成功",
                f"配置文件模板已创建于:\n{actual_config_file_path}\n\n请填入您的 GEMINI_API_KEY，并可根据需要修改各项设置（包括头像图片）后重新运行程序。",
            )
            return False
        except IOError as e:
            QMessageBox.critical(None, "错误", f"无法创建配置文件: {e}")
            return False
    try:
        config_manager_global = ConfigManager(config_file=config_file_relative_path)
    except FileNotFoundError as e:
        QMessageBox.critical(None, "配置错误", f"无法加载配置文件: {e}\n程序将退出。")
        return False
    except Exception as e:
        QMessageBox.critical(
            None, "配置错误", f"加载配置文件时发生未知错误: {e}\n程序将退出。"
        )
        return False
    initial_image_filename = config_manager_global.get_pet_initial_image_filename()
    initial_pet_image_abs_path = os.path.join(
        assets_path_global, initial_image_filename
    )
    if not os.path.exists(initial_pet_image_abs_path):
        print(
            f"警告：初始宠物图片 {initial_pet_image_abs_path} 不存在。正在尝试创建占位图片。"
        )
        create_placeholder_avatar(
            initial_pet_image_abs_path,
            "Pet",
            size=(120, 120),
            bg_color=(200, 200, 200, 180),
        )
    avatar_base_path_relative = config_manager_global.get_avatar_base_path_relative()
    avatar_base_path_global = os.path.normpath(
        os.path.join(project_root, avatar_base_path_relative)
    )
    os.makedirs(avatar_base_path_global, exist_ok=True)
    print(f"Avatar base path set to: {avatar_base_path_global}")
    pet_avatar_filename = config_manager_global.get_pet_avatar_filename()
    pet_avatar_path_global = os.path.join(avatar_base_path_global, pet_avatar_filename)
    if not os.path.exists(pet_avatar_path_global):
        print(f"警告：宠物头像图片 {pet_avatar_path_global} 不存在。")
        create_placeholder_avatar(pet_avatar_path_global, "P")
    user_avatar_filename = config_manager_global.get_user_avatar_filename()
    user_avatar_path_global = os.path.join(
        avatar_base_path_global, user_avatar_filename
    )
    if not os.path.exists(user_avatar_path_global):
        print(f"警告：用户头像图片 {user_avatar_path_global} 不存在。")
        create_placeholder_avatar(user_avatar_path_global, "U")
    http_proxy = config_manager_global.get_http_proxy()
    https_proxy = config_manager_global.get_https_proxy()
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
    return True


def initialize_services():
    global gemini_client_global, config_manager_global, mongo_handler_global
    api_key = config_manager_global.get_gemini_api_key()
    model_name = config_manager_global.get_gemini_model_name()
    pet_name = config_manager_global.get_pet_name()
    user_name = config_manager_global.get_user_name()
    pet_persona = config_manager_global.get_pet_persona()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        QMessageBox.critical(
            None, "API Key 错误", "请在 config/settings.ini 中配置您的 Gemini API Key。"
        )
        return False
    try:
        conn_str = config_manager_global.get_mongo_connection_string()
        db_name = config_manager_global.get_mongo_database_name()
        coll_name = config_manager_global.get_mongo_collection_name()
        mongo_handler_global = MongoHandler(conn_str, db_name, coll_name)
        if not mongo_handler_global.is_connected():
            print("警告: 无法连接到MongoDB，聊天记录功能将不可用。")
        else:
            print("MongoDB Handler 初始化并连接成功。")
    except Exception as e:
        print(f"警告: 初始化 MongoDB 时发生错误: {e}。聊天记录功能将不可用。")
        mongo_handler_global = None
    try:
        gemini_client_global = GeminiClient(
            api_key=api_key,
            model_name=model_name,
            pet_name=pet_name,
            user_name=user_name,
            pet_persona=pet_persona,
        )
        print("Gemini客户端初始化成功。")
    except ValueError as e:
        QMessageBox.critical(None, "Gemini客户端初始化错误", str(e))
        return False
    except ConnectionError as e:
        QMessageBox.critical(None, "Gemini连接错误", f"无法连接到Gemini服务: {e}")
        return False
    except Exception as e:
        QMessageBox.critical(
            None, "Gemini客户端错误", f"初始化Gemini客户端时发生未知错误: {e}"
        )
        return False
    return True


def open_chat_dialog_handler():
    global chat_dialog_global, gemini_client_global, pet_window_global, mongo_handler_global, config_manager_global
    global pet_avatar_path_global, user_avatar_path_global
    if chat_dialog_global is None:
        print("创建新的聊天对话框实例...")
        if not gemini_client_global:
            QMessageBox.warning(
                None, "服务未就绪", "Gemini 服务尚未初始化，无法打开聊天窗口。"
            )
            return
        chat_dialog_global = ChatDialog(
            gemini_client=gemini_client_global,
            mongo_handler=mongo_handler_global,
            config_manager=config_manager_global,
            pet_avatar_path=pet_avatar_path_global,
            user_avatar_path=user_avatar_path_global,
            parent=pet_window_global,
        )
        if pet_window_global:
            chat_dialog_global.emotion_received.connect(pet_window_global.set_emotion)
            print("已连接 ChatDialog.emotion_received 到 PetWindow.set_emotion")
    if chat_dialog_global.isHidden():
        print("显示聊天对话框...")
        chat_dialog_global.open_dialog()
    else:
        print("聊天对话框已显示，激活它...")
        chat_dialog_global.activateWindow()
        chat_dialog_global.raise_()


def main():
    global pet_window_global, config_manager_global, assets_path_global, mongo_handler_global
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    if not setup_environment_and_config():
        sys.exit(1)
    if not initialize_services():
        sys.exit(1)
    initial_image_filename = config_manager_global.get_pet_initial_image_filename()
    initial_pet_image_abs_path = os.path.join(
        assets_path_global, initial_image_filename
    )
    pet_window_global = PetWindow(
        initial_image_path=initial_pet_image_abs_path,
        assets_base_path=assets_path_global,
    )
    pet_window_global.request_open_chat_dialog.connect(open_chat_dialog_handler)
    pet_window_global.show()
    exit_code = app.exec()
    if mongo_handler_global:
        mongo_handler_global.close_connection()
    print("应用程序退出。")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
