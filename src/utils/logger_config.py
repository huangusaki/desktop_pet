"""
统一的日志配置模块
提供全局日志格式化、Uvicorn日志中文化和日志过滤功能
"""
import logging
import sys
from typing import Dict, Any
import coloredlogs


class UvicornFormatter(coloredlogs.ColoredFormatter):
    """自定义Uvicorn日志格式化器，实现中文化、格式统一和颜色输出"""
    
    # 消息翻译映射 - 仅翻译解释性语言，保留术语
    TRANSLATIONS: Dict[str, str] = {
        # Uvicorn 服务器消息
        "Started server process": "服务器进程已启动",
        "Waiting for application startup": "等待应用启动",
        "Application startup complete": "应用启动完成",
        "Uvicorn running on": "Uvicorn 运行在",
        "Press CTRL+C to quit": "按 CTRL+C 退出",
        "Shutting down": "正在关闭",
        "Finished server process": "服务器进程已结束",
        
        # WebSocket 消息
        "connection open": "连接已建立",
        "connection closed": "连接已关闭",
        "WebSocket": "WebSocket",
        "accepted": "已接受",
        
        # 应用上下文消息
        "ApplicationContext: AgentCore could not be imported.": "应用上下文: AgentCore 无法导入",
        "ApplicationContext initialization complete": "应用上下文初始化完成",
        "Starting web server...": "正在启动Web服务器...",
        "Creating services...": "正在创建服务...",
        "does not exist. A template will be created.": "不存在,将创建模板",
        "Config template created at": "配置模板已创建于",
        "Initializing ConfigManager...": "正在初始化配置管理器...",
        "ConfigManager initialized": "配置管理器已初始化",
        "Setting up avatar paths...": "正在设置头像路径...",
        "Initializing MongoHandler...": "正在初始化MongoDB处理器...",
        "MongoHandler connected": "MongoDB处理器已连接",
        "MongoHandler failed to connect": "MongoDB处理器连接失败",
        "Initializing RelationshipManager...": "正在初始化关系管理器...",
        "Initializing PromptBuilder...": "正在初始化提示构建器...",
        "Initializing GeminiClient...": "正在初始化Gemini客户端...",
        "GeminiClient initialized": "Gemini客户端已初始化",
        "Initializing AgentCore...": "正在初始化Agent核心...",
        "AgentCore initialized": "Agent核心已初始化",
        "Initializing HippocampusManager...": "正在初始化记忆系统管理器...",
        "Memory system (HippocampusManager) initialized successfully.": "记忆系统(HippocampusManager)初始化成功",
        "Preloading WebChatWindow...": "正在预加载Web聊天窗口...",
        "WebChatWindow preloaded successfully": "Web聊天窗口预加载成功",
        "Screen analysis feature is available.": "屏幕分析功能可用",
        "Screen analysis feature is unavailable": "屏幕分析功能不可用",
        "disabled in config or missing dependencies": "已在配置中禁用或缺少依赖",
        "Opening web chat window...": "正在打开Web聊天窗口...",
        "Starting Web server for chat interface...": "正在启动聊天界面Web服务器...",
        "Web server started on": "Web服务器已启动于",
        "You can also access from mobile:": "也可从移动设备访问:",
        "task scheduled to run every": "任务已计划每",
        "minutes.": "分钟运行一次",
        
        # Agent Core 消息
        "AgentCore initialized.": "Agent核心已初始化",
        "Agent mode set to:": "Agent模式已设置为:",
        
        # Screen Analyzer 消息
        "Pillow not found, screen analysis feature disabled.": "未找到Pillow库,屏幕分析功能已禁用",
        "Pillow not found and global TTS is disabled. ScreenAnalyzer will be largely inactive.": "未找到Pillow库且全局TTS已禁用,屏幕分析器将基本不活动",
        "Screen analysis monitoring enabled via external request.": "屏幕分析监控已通过外部请求启用",
        "Screen analysis monitoring disabled via external request.": "屏幕分析监控已通过外部请求禁用",
        "Screen analysis monitoring not started": "屏幕分析监控未启动",
        "feature disabled in config": "功能已在配置中禁用",
        "TTS can still be triggered by chat.": "TTS仍可通过聊天触发",
        "Cannot start screen analysis monitoring, Gemini client not available.": "无法启动屏幕分析监控,Gemini客户端不可用",
        "Analysis thread did not quit gracefully, terminating.": "分析线程未正常退出,正在终止",
        "TTS request thread did not quit gracefully, terminating.": "TTS请求线程未正常退出,正在终止",
        "Screen monitoring stopped, threads cleaned up, and TTS queue cleared.": "屏幕监控已停止,线程已清理,TTS队列已清空",
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # 先调用父类完成格式化（包括参数替换和颜色添加）
        formatted_message = super().format(record)
        
        # 然后翻译已格式化的消息（保留ANSI颜色代码）
        for en, zh in self.TRANSLATIONS.items():
            if en in formatted_message:
                formatted_message = formatted_message.replace(en, zh)
        
        return formatted_message


class HTTPAccessLogFilter(logging.Filter):
    """过滤HTTP访问日志，减少噪音"""
    
    def __init__(self, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
    
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        
        # 如果不是verbose模式，过滤掉静态资源请求和常见API
        if not self.verbose:
            # 过滤静态资源、常见API和GET请求
            static_patterns = [
                # 静态资源
                "/assets/",
                "/vite.svg",
                ".css",
                ".js",
                ".png",
                ".jpg",
                ".ico",
                ".woff",
                ".woff2",
                ".ttf",
                ".svg",
                
                # 常见API路径
                "/api/config",          # 配置查询
                "/api/chat/history",    # 聊天历史
                "/api/avatar/",         # 头像请求
                "/?platform=",          # 平台参数
                
                # 通用GET请求过滤(只保留POST/PUT/DELETE等重要请求)
                '"GET /',               # 所有GET请求
            ]
            for pattern in static_patterns:
                if pattern in msg:
                    return False
        
        return True


def setup_logging(access_log_verbose: bool = False) -> None:
    """
    配置全局日志系统
    
    Args:
        access_log_verbose: 是否显示详细的HTTP访问日志
    """
    # 统一的日志格式
    log_format = "%(asctime)s [%(name)s:%(lineno)d] %(levelname)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 1. 先用coloredlogs配置根日志器
    coloredlogs.install(
        level="INFO",
        fmt=log_format,
        datefmt=date_format,
        level_styles={
            "debug": {"color": "cyan"},
            "info": {"color": "green"},
            "warning": {"color": "yellow", "bold": True},
            "error": {"color": "red"},
            "critical": {"color": "red", "bold": True, "background": "white"},
        },
        field_styles={
            "asctime": {"color": "green"},
            "levelname": {"color": "green", "bold": True},
            "name": {"color": "blue"},
            "lineno": {"color": "blue"},
        },
    )
    
    # 2. 为Uvicorn loggers单独配置coloredlogs + 翻译
    uvicorn_loggers = [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ]
    
    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)
        
        # 先用coloredlogs为这个logger安装彩色输出
        coloredlogs.install(
            logger=logger,  # 指定logger
            level="INFO",
            fmt=log_format,
            datefmt=date_format,
            level_styles={
                "debug": {"color": "cyan"},
                "info": {"color": "green"},
                "warning": {"color": "yellow", "bold": True},
                "error": {"color": "red"},
                "critical": {"color": "red", "bold": True, "background": "white"},
            },
            field_styles={
                "asctime": {"color": "green"},
                "levelname": {"color": "green", "bold": True},
                "name": {"color": "blue"},
                "lineno": {"color": "blue"},
            },
        )
        
        # 然后为handler的formatter包装翻译功能
        if logger.handlers:
            original_formatter = logger.handlers[0].formatter
            
            # 创建翻译包装器
            class TranslatingFormatter(logging.Formatter):
                def __init__(self, base_formatter):
                    self.base_formatter = base_formatter
                
                def format(self, record):
                    # 先用原始formatter格式化（包含颜色）
                    formatted = self.base_formatter.format(record)
                    # 然后应用翻译
                    for en, zh in UvicornFormatter.TRANSLATIONS.items():
                        if en in formatted:
                            formatted = formatted.replace(en, zh)
                    return formatted
            
            # 替换formatter
            logger.handlers[0].setFormatter(TranslatingFormatter(original_formatter))
        
        # 为access日志添加过滤器
        if logger_name == "uvicorn.access" and logger.handlers:
            logger.handlers[0].addFilter(HTTPAccessLogFilter(verbose=access_log_verbose))
        
        # 防止日志传播到root logger（避免重复）
        logger.propagate = False
    
    # 降低Qt相关日志级别，减少噪音
    logging.getLogger("PyQt6").setLevel(logging.WARNING)
    logging.getLogger("qt").setLevel(logging.WARNING)
    
    logging.info("日志系统配置完成")


def get_uvicorn_log_config() -> Dict[str, Any]:
    """
    获取Uvicorn兼容的日志配置字典
    
    Returns:
        可直接传递给uvicorn.run()的log_config参数
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "src.utils.logger_config.UvicornFormatter",
                "fmt": "%(asctime)s [%(name)s:%(lineno)d] %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "level_styles": {
                    "debug": {"color": "cyan"},
                    "info": {"color": "green"},
                    "warning": {"color": "yellow", "bold": True},
                    "error": {"color": "red"},
                    "critical": {"color": "red", "bold": True, "background": "white"},
                },
                "field_styles": {
                    "asctime": {"color": "green"},
                    "levelname": {"color": "green", "bold": True},
                    "name": {"color": "blue"},
                    "lineno": {"color": "blue"},
                },
            },
        },
        "filters": {
            "access_filter": {
                "()": "src.utils.logger_config.HTTPAccessLogFilter",
                "verbose": False,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "filters": ["access_filter"],
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }
