"""
Web Chat Window using PyQt6 WebEngine.
Displays the web-based chat interface in a standalone PyQt window.
"""
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import QUrl, QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPainterPath, QColor
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile
import logging
import os

logger = logging.getLogger("WebChatWindow")


class WebChatWindow(QMainWindow):
    """Standalone window that embeds the web chat interface."""
    
    def __init__(self, url: str = "http://localhost:8765", parent=None, avatar_path: str = None):
        super().__init__(parent)
        self.setWindowTitle("Arisu Chat")
        self.resize(QSize(400, 600))
        
        # Let WebEngine handle its own rendering without Qt widget optimizations
        # Removing these attributes allows WebEngine to use its native refresh rate
        
        # Set window icon if avatar path is provided
        if avatar_path and os.path.exists(avatar_path):
            try:
                self._set_circular_icon(avatar_path)
            except Exception as e:
                logger.error(f"Failed to set window icon: {e}")
        
        # Configure WebEngine settings to reduce flickering
        profile = QWebEngineProfile.defaultProfile()
        settings = profile.settings()
        
        # Enable hardware acceleration
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        
        # Enable smooth scrolling animation
        settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)
        
        # Create web view
        # Lazy import to avoid startup freeze
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        self.web_view = QWebEngineView()
        
        # Let WebEngine use default rendering pipeline for optimal frame rate
        # No widget attributes needed - WebEngine has its own GPU-accelerated rendering
        
        # Set background color to avoid transparency-related flickering
        self.web_view.page().setBackgroundColor(QColor("#1a1c2c"))
        
        # Add platform=pyqt parameter to let frontend detect PyQt environment
        if '?' in url:
            url_with_param = f"{url}&platform=pyqt"
        else:
            url_with_param = f"{url}?platform=pyqt"
        
        self.web_view.setUrl(QUrl(url_with_param))
        self.setCentralWidget(self.web_view)
        
        logger.info(f"WebChatWindow 已初始化，URL: {url_with_param}")
    
    def _set_circular_icon(self, image_path: str):
        """Create a circular icon from the image path."""
        original_pixmap = QPixmap(image_path)
        if original_pixmap.isNull():
            return
            
        size = min(original_pixmap.width(), original_pixmap.height())
        target_size = QSize(size, size)
        
        circular_pixmap = QPixmap(target_size)
        circular_pixmap.fill(QColor("transparent"))
        
        painter = QPainter(circular_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)
        
        painter.drawPixmap(0, 0, original_pixmap)
        painter.end()
        
        self.setWindowIcon(QIcon(circular_pixmap))

    def closeEvent(self, event):
        """Handle window close event."""
        # Instead of closing, just hide the window to keep the session alive
        logger.info("WebChatWindow 正在隐藏")
        self.hide()
        event.ignore()
