"""
Configuration schema definitions for WebUI configuration management.
Defines all config items with their types, validation rules, and categories.
"""
from enum import Enum
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field


class ConfigCategory(str, Enum):
    """Configuration categories for organizing settings."""
    BASIC = "basic"  # 基础设置
    API = "api"  # API配置
    DATABASE = "database"  # 数据库
    MEMORY = "memory"  # 记忆系统
    SCREEN = "screen"  # 屏幕分析
    TTS = "tts"  # 语音合成
    AGENT = "agent"  # 智能体模式
    AVATARS = "avatars"  # 头像设置


class ConfigItem(BaseModel):
    """Individual configuration item."""
    section: str  # INI section name
    key: str  # INI key name
    value: Any  # Current value
    value_type: str  # 'string', 'int', 'float', 'bool', 'enum'
    default_value: Any  # Default value
    label: str  # Display label
    description: str  # Description for users
    category: ConfigCategory  # Category this item belongs to
    required: bool = False  # Whether this field is required
    sensitive: bool = False  # Whether this is sensitive data (like API keys)
    options: Optional[List[str]] = None  # Available options for enum types


# Config schema definitions by category
CONFIG_SCHEMA: Dict[ConfigCategory, List[Dict[str, Any]]] = {
    ConfigCategory.BASIC: [
        {
            "section": "BOT",
            "key": "NAME",
            "default": "小助手",
            "type": "string",
            "label": "Bot名称",
            "description": "您的桌面Bot的名字",
            "required": True
        },
        {
            "section": "USER",
            "key": "NAME",
            "default": "主人",
            "type": "string",
            "label": "用户名称",
            "description": "您的名字,Bot会这样称呼您",
            "required": True
        },
        {
            "section": "BOT",
            "key": "PERSONA",
            "default": "你是一个友好、乐于助人的桌面Bot。",
            "type": "text",
            "label": "Bot人格",
            "description": "定义Bot的性格和行为方式",
            "required": True
        },
        {
            "section": "BOT",
            "key": "INITIAL_IMAGE_FILENAME",
            "default": "default.png",
            "type": "string",
            "label": "初始图片文件名",
            "description": "Bot窗口的初始显示图片",
            "required": False
        },
        {
            "section": "BOT",
            "key": "AGENT_MODE_EMOTIONS",
            "default": "'neutral', 'focused', 'helpful'",
            "type": "string",
            "label": "Agent模式情绪",
            "description": "智能体模式下可用的情绪表情",
            "required": False
        },
        {
            "section": "BOT",
            "key": "SPEECH_PATTERN",
            "default": '"哈……?你是笨蛋吗?"\n"爱丽丝死了哦,现在是作为地缚灵在说话"\n"爱丽丝的心很大,可以装下很多喜欢的东西"\n"因为是前辈的请求,爱丽丝才会答应的,前辈可不要忘了这点哦?"',
            "type": "text",
            "label": "说话风格示例",
            "description": "Bot的说话风格参考示例,用于指导对话生成时的语气和表达方式",
            "required": False
        },
        {
            "section": "BOT",
            "key": "CONSTRAINTS",
            "default": """【表达规则】
1、严格遵守json格式输出规范。
2、检查你之前的发言,避免内容和措辞上的重复,包括语气词"嗯"、"哼"、"呢"、"呀"、"呵"等也要尽可能地少用。
3、不要刻意突出人设,不要过度纠结于自己的人设,严禁提及任何设定里的内容,禁止发送emoji或者表情。
4、注意聊天记录的时间,注意中文标点符号正确使用方式,比如省略号要用"……"而不是"...",也不要弄得全都是省略号,禁止使用"'"和"'"符号,表达方式要丰富,不要总是把回复重心放在{user_name},断句要合理。
5、表达情绪多样化,基于对话内容,会害羞、生气、愤怒、友善、依赖等,具体要体现在回复内容以及返回表情,不要盲目乐观。
6、不要拓展到无关话题上,三观要正,对不合适的事要指责。
7、回答要口语化、日常化,不要书面化,应该更要有人和人对话的感觉。
8、text_japanese中,请将{bot_name}转成片假名,不要使用原名""",
            "type": "text",
            "label": "表达规则约束",
            "description": "Bot对话时需要遵守的表达规则和约束条件",
            "required": False
        },
        {
            "section": "BOT",
            "key": "FORMAT_EXAMPLE",
            "default": """{
  "text": "你好~我是{bot_name}哦!",
  "emotion": "{default_emotion}",
  "tone": "{default_tone}",
  "favorability_change": 1,
  "text_japanese": "こんにちは、{bot_name}です!",
  "emotion_update": {
    "state": "嬉しい",
    "reason": "優しく挨拶してくれたから"
  }
}""",
            "type": "text",
            "label": "JSON格式示例",
            "description": "Bot回复的JSON格式示例,用于指导输出格式",
            "required": False
        }
    ],
    
    ConfigCategory.API: [
        {
            "section": "API",
            "key": "PRIMARY_LLM_PROVIDER",
            "default": "gemini",
            "type": "enum",
            "options": ["gemini", "openai"],
            "label": "主要LLM提供商",
            "description": "选择主要使用的LLM提供商",
            "required": True
        },
        {
            "section": "GEMINI",
            "key": "API_KEY",
            "default": "",
            "type": "password",
            "label": "Gemini API密钥",
            "description": "Google Gemini API的密钥",
            "required": True,
            "sensitive": True
        },
        {
            "section": "GEMINI",
            "key": "MODEL_NAME",
            "default": "gemini-1.5-flash-latest",
            "type": "string",
            "label": "模型名称",
            "description": "使用的Gemini模型名称",
            "required": True
        },
        {
            "section": "GEMINI",
            "key": "HTTP_PROXY",
            "default": "",
            "type": "string",
            "label": "HTTP代理",
            "description": "HTTP代理服务器地址(可选)",
            "required": False
        },
        {
            "section": "GEMINI",
            "key": "HTTPS_PROXY",
            "default": "",
            "type": "string",
            "label": "HTTPS代理",
            "description": "HTTPS代理服务器地址(可选)",
            "required": False
        },
        # OpenAI 配置
        {
            "section": "OPENAI",
            "key": "ENABLED",
            "default": False,
            "type": "bool",
            "label": "启用OpenAI",
            "description": "是否启用OpenAI API支持",
            "required": False
        },
        {
            "section": "OPENAI",
            "key": "API_KEY",
            "default": "",
            "type": "password",
            "label": "OpenAI API密钥",
            "description": "OpenAI API的密钥(或兼容API的密钥)",
            "required": False,
            "sensitive": True
        },
        {
            "section": "OPENAI",
            "key": "MODEL_NAME",
            "default": "gpt-3.5-turbo",
            "type": "string",
            "label": "模型名称",
            "description": "使用的OpenAI模型名称(如 gpt-4, gpt-3.5-turbo)",
            "required": False
        },
        {
            "section": "OPENAI",
            "key": "BASE_URL",
            "default": "https://api.openai.com/v1",
            "type": "string",
            "label": "API基础URL",
            "description": "OpenAI API的基础URL(可用于兼容的第三方API)",
            "required": False
        },
        {
            "section": "OPENAI",
            "key": "TEMPERATURE",
            "default": 0.7,
            "type": "float",
            "label": "温度参数",
            "description": "控制生成文本的随机性(0-2之间,越高越随机)",
            "required": False
        },
        {
            "section": "OPENAI",
            "key": "MAX_TOKENS",
            "default": 1000,
            "type": "int",
            "label": "最大Token数",
            "description": "生成文本的最大token数量",
            "required": False
        },
        {
            "section": "OPENAI",
            "key": "TIMEOUT_SECONDS",
            "default": 30,
            "type": "int",
            "label": "请求超时(秒)",
            "description": "API请求的超时时间",
            "required": False
        }
    ],
    
    ConfigCategory.DATABASE: [
        {
            "section": "MONGODB",
            "key": "CONNECTION_STRING",
            "default": "mongodb://localhost:27017/",
            "type": "string",
            "label": "MongoDB连接字符串",
            "description": "MongoDB数据库的连接URL",
            "required": True,
            "sensitive": True
        },
        {
            "section": "MONGODB",
            "key": "DATABASE_NAME",
            "default": "desktop_bot_db",
            "type": "string",
            "label": "数据库名称",
            "description": "使用的数据库名称",
            "required": True
        },
        {
            "section": "MONGODB",
            "key": "COLLECTION_NAME",
            "default": "chat_history",
            "type": "string",
            "label": "集合名称",
            "description": "聊天历史存储的集合名称",
            "required": True
        },
        {
            "section": "MONGODB",
            "key": "HISTORY_COUNT_FOR_PROMPT",
            "default": 5,
            "type": "int",
            "label": "历史消息数量",
            "description": "构建提示词时使用的历史消息数量",
            "required": False
        },
        {
            "section": "MONGODB",
            "key": "CHAT_DIALOG_DISPLAY_HISTORY_COUNT",
            "default": 0,
            "type": "int",
            "label": "对话框历史显示数",
            "description": "聊天对话框中显示的历史消息数量(0为不限制)",
            "required": False
        }
    ],
    
    ConfigCategory.MEMORY: [
        {
            "section": "MEMORY_SYSTEM",
            "key": "BUILD_INTERVAL_SECONDS",
            "default": 600,
            "type": "int",
            "label": "记忆构建间隔(秒)",
            "description": "自动构建记忆的时间间隔",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "FORGET_INTERVAL_SECONDS",
            "default": 3600,
            "type": "int",
            "label": "记忆遗忘间隔(秒)",
            "description": "自动遗忘记忆的时间间隔",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "CONSOLIDATE_INTERVAL_SECONDS",
            "default": 3600,
            "type": "int",
            "label": "记忆整合间隔(秒)",
            "description": "自动整合记忆的时间间隔",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "RUN_CONSOLIDATE_ON_STARTUP",
            "default": True,
            "type": "bool",
            "label": "启动时整合记忆",
            "description": "应用启动时是否立即执行记忆整合",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "RUN_BUILD_ON_STARTUP_AFTER_CONSOLIDATE",
            "default": True,
            "type": "bool",
            "label": "整合后构建记忆",
            "description": "整合完成后是否立即构建记忆",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "BUILD_DISTRIBUTION",
            "default": "3.0,2.0,0.5,72.0,24.0,0.5",
            "type": "string",
            "label": "记忆构建分布",
            "description": "记忆构建的时间分布参数(逗号分隔)",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "BUILD_SAMPLE_NUM",
            "default": 5,
            "type": "int",
            "label": "采样数量",
            "description": "记忆构建时的采样数量",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "BUILD_SAMPLE_LENGTH",
            "default": 10,
            "type": "int",
            "label": "采样长度",
            "description": "记忆构建时的采样长度",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "COMPRESS_RATE",
            "default": 0.08,
            "type": "float",
            "label": "压缩率",
            "description": "记忆压缩的比率",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "FORGET_TIME_HOURS",
            "default": 48.0,
            "type": "float",
            "label": "遗忘时间(小时)",
            "description": "记忆的遗忘时间阈值",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "FORGET_PERCENTAGE",
            "default": 0.005,
            "type": "float",
            "label": "遗忘百分比",
            "description": "每次遗忘操作的百分比",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "BAN_WORDS",
            "default": "",
            "type": "string",
            "label": "禁用词列表",
            "description": "记忆系统中需要过滤的词语(逗号分隔)",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "CONSOLIDATE_PERCENTAGE",
            "default": 0.1,
            "type": "float",
            "label": "整合百分比",
            "description": "记忆整合的百分比",
            "required": False
        },
        {
            "section": "MEMORY_SYSTEM",
            "key": "CONSOLIDATION_SIMILARITY_THRESHOLD",
            "default": 0.90,
            "type": "float",
            "label": "整合相似度阈值",
            "description": "判断记忆是否需要整合的相似度阈值",
            "required": False
        }
    ],
    
    ConfigCategory.SCREEN: [
        {
            "section": "SCREEN_ANALYSIS",
            "key": "ENABLED",
            "default": False,
            "type": "bool",
            "label": "启用屏幕分析",
            "description": "是否启用自动屏幕分析功能",
            "required": False
        },
        {
            "section": "SCREEN_ANALYSIS",
            "key": "TASK_TIMEOUT_SECONDS",
            "default": 60,
            "type": "int",
            "label": "任务超时(秒)",
            "description": "屏幕分析任务的超时时间",
            "required": False
        },
        {
            "section": "SCREEN_ANALYSIS",
            "key": "MIN_INTERVAL_SECONDS",
            "default": 60,
            "type": "int",
            "label": "最小间隔(秒)",
            "description": "两次屏幕分析之间的最小时间间隔",
            "required": False
        },
        {
            "section": "SCREEN_ANALYSIS",
            "key": "MAX_INTERVAL_SECONDS",
            "default": 300,
            "type": "int",
            "label": "最大间隔(秒)",
            "description": "两次屏幕分析之间的最大时间间隔",
            "required": False
        },
        {
            "section": "SCREEN_ANALYSIS",
            "key": "CHANCE",
            "default": 0.1,
            "type": "float",
            "label": "触发概率",
            "description": "屏幕分析的触发概率(0-1之间)",
            "required": False
        },
        {
            "section": "SCREEN_ANALYSIS",
            "key": "SAVE_REACTION_TO_CHAT_HISTORY",
            "default": True,
            "type": "bool",
            "label": "保存到聊天历史",
            "description": "是否将屏幕分析结果保存到聊天历史",
            "required": False
        },
        {
            "section": "SCREEN_ANALYSIS",
            "key": "PROMPT",
            "default": "你是{bot_name}，一个可爱的桌面Bot。这张图片是用户当前的屏幕截图。\n请根据屏幕内容，用你的角色口吻，简短地、不经意地发表一句评论或感想。\n你的回复必须是一个JSON对象，包含 'text' (你作为Bot说的话，字符串) 和 'emotion' (你当前的情绪，从 {available_emotions_str} 中选择一个，字符串)。",
            "type": "text",
            "label": "分析提示词",
            "description": "屏幕分析时使用的提示词模板",
            "required": False
        }
    ],
    
    ConfigCategory.TTS: [
        {
            "section": "TTS",
            "key": "ENABLED",
            "default": False,
            "type": "bool",
            "label": "启用TTS",
            "description": "是否启用语音合成功能",
            "required": False
        },
        {
            "section": "TTS",
            "key": "API_BASE_URL",
            "default": "http://127.0.0.1:9880/",
            "type": "string",
            "label": "TTS API地址",
            "description": "TTS服务的API基础URL",
            "required": False
        },
        {
            "section": "TTS",
            "key": "API_ENDPOINT",
            "default": "tts",
            "type": "string",
            "label": "API端点",
            "description": "TTS API的端点路径",
            "required": False
        },
        {
            "section": "TTS",
            "key": "AVAILABLE_TONES",
            "default": "normal",
            "type": "string",
            "label": "可用语调",
            "description": "可用的语调列表(逗号分隔)",
            "required": False
        },
        {
            "section": "TTS",
            "key": "DEFAULT_TONE",
            "default": "normal",
            "type": "string",
            "label": "默认语调",
            "description": "默认使用的语调",
            "required": False
        },
        {
            "section": "TTS",
            "key": "PROMPT_LANGUAGE",
            "default": "zh",
            "type": "string",
            "label": "提示语言",
            "description": "TTS提示词的语言",
            "required": False
        },
        {
            "section": "TTS",
            "key": "TEXT_LANGUAGE",
            "default": "zh",
            "type": "string",
            "label": "文本语言",
            "description": "TTS合成文本的语言",
            "required": False
        },
        {
            "section": "TTS",
            "key": "CUT_PUNC_METHOD",
            "default": "cut5",
            "type": "string",
            "label": "分句方法",
            "description": "文本分句的方法",
            "required": False
        },
        {
            "section": "TTS",
            "key": "MEDIA_TYPE",
            "default": "wav",
            "type": "string",
            "label": "音频格式",
            "description": "生成音频的格式",
            "required": False
        },
        {
            "section": "TTS",
            "key": "PLAY_AUDIO_TIMEOUT_SECONDS",
            "default": 45,
            "type": "int",
            "label": "播放超时(秒)",
            "description": "音频播放的超时时间",
            "required": False
        }
    ],
    
    ConfigCategory.AGENT: [
        {
            "section": "AGENT",
            "key": "PYAUTOGUI_PAUSE",
            "default": 0.25,
            "type": "float",
            "label": "PyAutoGUI暂停",
            "description": "PyAutoGUI操作之间的暂停时间(秒)",
            "required": False
        },
        {
            "section": "AGENT",
            "key": "ACTIVE_WINDOW_DELAY_BEFORE_TYPE",
            "default": 0.75,
            "type": "float",
            "label": "输入前延迟",
            "description": "激活窗口后输入前的延迟时间(秒)",
            "required": False
        },
        {
            "section": "AGENT",
            "key": "GET_WINDOW_TITLE_DELAY",
            "default": 0.3,
            "type": "float",
            "label": "获取窗口标题延迟",
            "description": "获取窗口标题的延迟时间(秒)",
            "required": False
        },
        {
            "section": "AGENT",
            "key": "MAX_READ_FILE_LENGTH",
            "default": 2000,
            "type": "int",
            "label": "最大读取文件长度",
            "description": "Agent读取文件的最大长度(字符)",
            "required": False
        },
        {
            "section": "AGENT",
            "key": "STEP_DELAY_SECONDS",
            "default": 0.5,
            "type": "float",
            "label": "步骤延迟(秒)",
            "description": "Agent执行步骤之间的延迟时间",
            "required": False
        }
    ],
    
    ConfigCategory.AVATARS: [
        {
            "section": "AVATARS",
            "key": "AVATAR_BASE_PATH",
            "default": "src/assets/icon",
            "type": "string",
            "label": "头像基础路径",
            "description": "头像文件的基础目录路径",
            "required": False
        },
        {
            "section": "AVATARS",
            "key": "BOT_AVATAR_FILENAME",
            "default": "bot.png",
            "type": "string",
            "label": "Bot头像文件名",
            "description": "Bot头像的文件名",
            "required": False
        },
        {
            "section": "AVATARS",
            "key": "USER_AVATAR_FILENAME",
            "default": "user.png",
            "type": "string",
            "label": "用户头像文件名",
            "description": "用户头像的文件名",
            "required": False
        }
    ]
}
