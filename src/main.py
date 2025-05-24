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
from src.core.screen_analyzer import ScreenAnalyzer

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image, ImageDraw, ImageFont = None, None, None
    print(
        "CRITICAL: Pillow library not found. Image generation and screen analysis will fail. "
        "Please install it using 'pip install Pillow'"
    )
from typing import List

config_manager_global = None
gemini_client_global = None
pet_window_global = None
chat_dialog_global = None
assets_path_global = None
mongo_handler_global = None
avatar_base_path_global = None
pet_avatar_path_global = None
user_avatar_path_global = None
available_emotions_global: List[str] = ["default"]
screen_analyzer_global = None


def create_placeholder_avatar(
    image_path: str, text: str, size=(64, 64), bg_color=(128, 128, 128, 200)
):
    if not Image:
        print(f"Pillow not available, cannot create placeholder for {image_path}")
        return
    try:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img = Image.new("RGBA", size, bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font_size = size[1] // 2
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        elif hasattr(draw, "textsize"):
            text_width, text_height = draw.textsize(text, font=font)
        else:
            text_width, text_height = font_size * len(text) / 1.5, font_size
        x = (size[0] - text_width) / 2
        y = (size[1] - text_height) / 2
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
        img.save(image_path)
        print(f"已创建占位头像图片: {image_path}")
    except Exception as e:
        print(f"为 {image_path} 创建占位头像图片失败: {e}")


def scan_and_update_available_emotions(assets_path: str):
    global available_emotions_global
    if not os.path.isdir(assets_path):
        print(
            f"错误: 资源路径 {assets_path} 不是一个有效的目录。情绪列表将使用默认值。"
        )
        available_emotions_global = ["default"]
        return
    found_emotions = set()
    for filename in os.listdir(assets_path):
        if filename.lower().endswith(".png"):
            emotion_name = os.path.splitext(filename)[0].lower()
            found_emotions.add(emotion_name)
    if not found_emotions:
        print(
            f"警告: 在 {assets_path} 中未找到任何 .png 文件作为情绪。将使用 'default'。"
        )
        found_emotions.add("default")
    if "default" not in found_emotions:
        print(f"警告: 'default.png' 未在 {assets_path} 中找到。建议添加一个作为后备。")
        found_emotions.add("default")
    available_emotions_global = sorted(list(found_emotions))
    if not available_emotions_global:
        available_emotions_global = ["default"]
    print(f"可用的情绪列表已更新: {available_emotions_global}")


def setup_environment_and_config():
    global config_manager_global, assets_path_global, project_root
    global avatar_base_path_global, pet_avatar_path_global, user_avatar_path_global
    global available_emotions_global
    os.makedirs(os.path.join(project_root, "config"), exist_ok=True)
    assets_path_global = os.path.normpath(os.path.join(project_root, "src", "assets"))
    os.makedirs(assets_path_global, exist_ok=True)
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    scan_and_update_available_emotions(assets_path_global)
    config_file_relative_path = os.path.join("config", "settings.ini")
    actual_config_file_path = os.path.join(project_root, config_file_relative_path)
    if not os.path.exists(actual_config_file_path):
        print(f"警告：配置文件 {actual_config_file_path} 不存在。将创建一个模板。")
        try:
            with open(actual_config_file_path, "w", encoding="utf-8") as cf:
                cf.write("[GEMINI]\n")
                cf.write("API_KEY = YOUR_API_KEY_HERE\n")
                cf.write("MODEL_NAME = gemini-1.5-flash-latest\n")
                cf.write("HTTP_PROXY =\n")
                cf.write("HTTPS_PROXY =\n\n")
                cf.write("[PET]\n")
                cf.write("INITIAL_IMAGE_FILENAME = default.png\n")
                cf.write("NAME = 小助手\n")
                cf.write(
                    "PERSONA = 你是一个友好、乐于助人的桌面宠物，名叫“小助手”。你会用亲切的语气和用户交流。\n\n"
                )
                cf.write("[USER]\n")
                cf.write("NAME = 主人\n\n")
                cf.write("[AVATARS]\n")
                cf.write("AVATAR_BASE_PATH = src/assets/icon\n")
                cf.write("PET_AVATAR_FILENAME = bot.png\n")
                cf.write("USER_AVATAR_FILENAME = user.png\n\n")
                cf.write("[MONGODB]\n")
                cf.write("CONNECTION_STRING = mongodb://localhost:27017/\n")
                cf.write("DATABASE_NAME = desktop_pet_db\n")
                cf.write("COLLECTION_NAME = chat_history\n")
                cf.write("HISTORY_COUNT_FOR_PROMPT = 5\n\n")
                cf.write("[SCREEN_ANALYSIS]\n")
                cf.write("# 是否启用屏幕截图分析功能 (True/False)\n")
                cf.write("ENABLED = False\n")
                cf.write("# 屏幕分析触发检查的间隔时间（秒）。\n")
                cf.write("INTERVAL_SECONDS = 60\n")
                cf.write("# 每次检查时，实际执行屏幕分析的概率 (0.0 到 1.0)。\n")
                cf.write("CHANCE = 0.1\n")
                cf.write("# 发送给 LLM 的提示模板。\n")
                cf.write(
                    "# {pet_name}, {user_name}, {available_emotions_str} 会被替换。\n"
                )
                default_screen_prompt = (
                    "你是{pet_name}，一个可爱的桌面宠物。这张图片是你的主人 {user_name} 当前的屏幕截图。\\n"
                    "请根据屏幕内容，用你的角色口吻，对 {user_name} 不经意地发表一句评论或感想。\\n"
                    "重要：即使屏幕内容看起来和之前相似，也请你努力想出一些新的、不同的评论。\\n"
                    "你的回复必须是一个JSON对象，包含 'text' (你作为宠物说的话，字符串) 和 'emotion' (你当前的情绪，从 {available_emotions_str} 中选择一个，字符串)。\\n"
                    '例如：{{\\"text\\": \\"{user_name} 看的这个视频很有趣呢！\\", \\"emotion\\": \\"happy\\"}}'
                )
                cf.write(f"PROMPT = {default_screen_prompt}\n\n")
            QMessageBox.information(
                None,
                "配置文件创建成功",
                f"配置文件模板已创建于:\n{actual_config_file_path}\n\n请配置后重新运行。",
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
        print(f"警告：初始宠物图片 {initial_pet_image_abs_path} 不存在。创建占位图。")
        create_placeholder_avatar(initial_pet_image_abs_path, "Pet", size=(120, 120))
        if (
            initial_image_filename.lower() == "default.png"
            and "default" not in available_emotions_global
        ):
            available_emotions_global.append("default")
            available_emotions_global = sorted(list(set(available_emotions_global)))
    avatar_base_path_relative = config_manager_global.get_avatar_base_path_relative()
    avatar_base_path_global = os.path.normpath(
        os.path.join(project_root, avatar_base_path_relative)
    )
    os.makedirs(avatar_base_path_global, exist_ok=True)
    pet_avatar_filename = config_manager_global.get_pet_avatar_filename()
    pet_avatar_path_global = os.path.join(avatar_base_path_global, pet_avatar_filename)
    if not os.path.exists(pet_avatar_path_global):
        create_placeholder_avatar(pet_avatar_path_global, "P")
    user_avatar_filename = config_manager_global.get_user_avatar_filename()
    user_avatar_path_global = os.path.join(
        avatar_base_path_global, user_avatar_filename
    )
    if not os.path.exists(user_avatar_path_global):
        create_placeholder_avatar(user_avatar_path_global, "U")
    http_proxy = config_manager_global.get_http_proxy()
    https_proxy = config_manager_global.get_https_proxy()
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
    return True


def initialize_services():
    global gemini_client_global, config_manager_global, mongo_handler_global, available_emotions_global
    api_key = config_manager_global.get_gemini_api_key()
    model_name = config_manager_global.get_gemini_model_name()
    pet_name = config_manager_global.get_pet_name()
    user_name = config_manager_global.get_user_name()
    pet_persona = config_manager_global.get_pet_persona()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        QMessageBox.critical(
            None, "API Key 错误", "请在 config/settings.ini 中配置 Gemini API Key。"
        )
        return False
    try:
        conn_str = config_manager_global.get_mongo_connection_string()
        db_name = config_manager_global.get_mongo_database_name()
        coll_name = config_manager_global.get_mongo_collection_name()
        mongo_handler_global = MongoHandler(conn_str, db_name, coll_name)
        if not mongo_handler_global.is_connected():
            QMessageBox.warning(
                None, "MongoDB连接警告", "无法连接到MongoDB，聊天记录功能将不可用。"
            )
            mongo_handler_global = None
        else:
            print("MongoDB Handler 初始化并连接成功。")
    except Exception as e:
        QMessageBox.warning(
            None,
            "MongoDB初始化警告",
            f"初始化 MongoDB 时发生错误: {e}。\n聊天记录功能将不可用。",
        )
        mongo_handler_global = None
    try:
        gemini_client_global = GeminiClient(
            api_key=api_key,
            model_name=model_name,
            pet_name=pet_name,
            user_name=user_name,
            pet_persona=pet_persona,
            available_emotions=available_emotions_global,
        )
        print("Gemini客户端初始化成功。")
    except Exception as e:
        QMessageBox.critical(None, "Gemini客户端初始化错误", f"错误: {e}")
        return False
    return True


def open_chat_dialog_handler():
    global chat_dialog_global, gemini_client_global, pet_window_global, mongo_handler_global
    global config_manager_global, pet_avatar_path_global, user_avatar_path_global
    if chat_dialog_global is None:
        if not gemini_client_global:
            QMessageBox.warning(None, "服务未就绪", "Gemini 服务尚未初始化。")
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
            chat_dialog_global.speech_and_emotion_received.connect(
                pet_window_global.update_speech_and_emotion
            )
    if chat_dialog_global.isHidden():
        chat_dialog_global.open_dialog()
    else:
        chat_dialog_global.activateWindow()
        chat_dialog_global.raise_()


def handle_screen_analysis_reaction(text: str, emotion: str):
    global pet_window_global, chat_dialog_global, mongo_handler_global, config_manager_global
    if pet_window_global:
        pet_window_global.update_speech_and_emotion(text, emotion)
    if chat_dialog_global:
        pet_name = (
            config_manager_global.get_pet_name() if config_manager_global else "宠物"
        )
        display_text = f"（看了一眼屏幕）{text}"
        chat_dialog_global._add_message_to_display(
            sender_name_for_log_only=pet_name, message=display_text, is_user=False
        )
        print(f"Main: Screen reaction ('{text}') added to ChatDialog display.")
    if mongo_handler_global and mongo_handler_global.is_connected():
        current_pet_character = (
            config_manager_global.get_pet_name()
            if config_manager_global
            else "DefaultPet"
        )
        db_text = f"（看了一眼屏幕）{text}"
        mongo_handler_global.insert_chat_message(
            sender="pet",
            message_text=db_text,
            role_play_character=current_pet_character,
        )
        print(f"Main: Screen reaction ('{text}') saved to MongoDB.")
    elif mongo_handler_global:
        print("Main: MongoDB not connected. Screen reaction not saved.")
    else:
        print("Main: MongoDB not available. Screen reaction not saved.")


def main():
    global pet_window_global, config_manager_global, assets_path_global, mongo_handler_global
    global available_emotions_global, gemini_client_global, screen_analyzer_global
    if Image is None:
        QMessageBox.critical(
            None,
            "依赖缺失",
            "Pillow 库未找到或无法导入。\n程序无法运行。请安装 Pillow: pip install Pillow",
        )
        sys.exit(1)
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
        available_emotions=available_emotions_global,
    )
    pet_window_global.request_open_chat_dialog.connect(open_chat_dialog_handler)
    if gemini_client_global and config_manager_global and pet_window_global:
        user_name_for_analyzer = config_manager_global.get_user_name()
        screen_analyzer_global = ScreenAnalyzer(
            gemini_client=gemini_client_global,
            config_manager=config_manager_global,
            pet_window=pet_window_global,
            pet_name=config_manager_global.get_pet_name(),
            user_name=user_name_for_analyzer,
            available_emotions=available_emotions_global,
            parent=app,
        )
        screen_analyzer_global.pet_reaction_ready.connect(
            handle_screen_analysis_reaction
        )
        screen_analyzer_global.start_monitoring()
    else:
        print(
            "ScreenAnalyzer 未初始化，依赖项 (Gemini, Config, PetWindow) 未完全就绪。"
        )
    pet_window_global.show()
    pet_window_global.set_speech_text("你好！我在这里哦！")
    exit_code = app.exec()
    if screen_analyzer_global:
        screen_analyzer_global.stop_monitoring()
        print("ScreenAnalyzer监控已停止。")
    if mongo_handler_global:
        mongo_handler_global.close_connection()
        print("MongoDB连接已关闭。")
    print("应用程序退出。")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
