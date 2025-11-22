"""
Web server for chat interface using FastAPI.
Provides REST API and WebSocket endpoints for the web-based chat UI.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging
import os
import sys
from datetime import datetime
import mimetypes
import uuid
from src.utils.config_service import ConfigService
from src.utils.config_schema import ConfigItem
from src.utils.preset_models import PresetManager, Preset

logger = logging.getLogger("ChatWebServer")


# Request/Response models
class MessageRequest(BaseModel):
    text: str
    files: List[str] = []


class MessageResponse(BaseModel):
    status: str
    message_id: str


class AgentToggleRequest(BaseModel):
    enabled: bool


class ConfigResponse(BaseModel):
    bot_name: str
    user_name: str
    bot_avatar: str
    user_avatar: str
    agent_mode_enabled: bool
    available_emotions: List[str]


class ChatMessage(BaseModel):
    id: str
    sender: str
    text: str
    timestamp: str
    is_user: bool
    emotion: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, Any]


class ConfigSaveResponse(BaseModel):
    success: bool
    message: str
    errors: Optional[Dict[str, str]] = None


class PresetCreateRequest(BaseModel):
    """创建预设请求"""
    name: str
    bot_name: str
    bot_persona: str
    speech_pattern: str = ""
    constraints: str = ""
    format_example: str = ""
    avatar_filename: str = "default_preset.png"


class PresetUpdateRequest(BaseModel):
    """更新预设请求"""
    name: Optional[str] = None
    bot_name: Optional[str] = None
    bot_persona: Optional[str] = None
    speech_pattern: Optional[str] = None
    constraints: Optional[str] = None
    format_example: Optional[str] = None
    avatar_filename: Optional[str] = None


class ChatWebServer:
    """FastAPI web server for chat interface."""
    
    def __init__(self, services):
        """
        Initialize the web server with application services.
        
        Args:
            services: AppServices instance containing all backend services
        """
        self.app = FastAPI(title="Arisu Chat API", version="1.0.0")
        self.services = services
        self.active_connections: List[WebSocket] = []
        self.upload_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Initialize config service
        self.config_service = ConfigService(services.config_manager)
        
        # Initialize preset manager with config_manager for default preset creation
        project_root = services.config_manager.get_project_root()
        presets_dir = os.path.join(project_root, "data", "presets")
        avatars_dir = os.path.join(project_root, "data", "presets", "avatars")
        self.preset_manager = PresetManager(
            presets_dir=presets_dir, 
            avatars_dir=avatars_dir,
            config_manager=services.config_manager
        )
        
        self._setup_cors()
        self._setup_routes()
        
        logger.info("ChatWebServer 已初始化")

    
    def _setup_cors(self):
        """Configure CORS to allow access from browsers (including mobile)."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins (for mobile access)
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS 中间件已配置")
    
    def _setup_routes(self):
        """Set up all API routes and WebSocket endpoints."""
        
        # Serve static files (frontend build)
        frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
        if os.path.exists(frontend_dist):
            self.app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
            
            @self.app.get("/")
            async def serve_frontend():
                index_path = os.path.join(frontend_dist, "index.html")
                if os.path.exists(index_path):
                    return FileResponse(index_path)
                return {"message": "Frontend not built yet. Run: cd frontend && npm run build"}
            
            # Serve vite.svg and other root static files
            @self.app.get("/vite.svg")
            async def serve_vite_svg():
                svg_path = os.path.join(frontend_dist, "vite.svg")
                if os.path.exists(svg_path):
                    return FileResponse(svg_path)
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="vite.svg not found")
        
        # API Routes
        @self.app.get("/api/config", response_model=ConfigResponse)
        async def get_config():
            """Get application configuration."""
            return ConfigResponse(
                bot_name=self.services.config_manager.get_bot_name(),
                user_name=self.services.config_manager.get_user_name(),
                bot_avatar="/api/avatar/bot",
                user_avatar="/api/avatar/user",
                agent_mode_enabled=self.services.agent_core.is_agent_mode_active if self.services.agent_core else False,
                available_emotions=self.services.available_emotions
            )
        
        @self.app.get("/api/chat/history")
        async def get_chat_history(limit: int = 20):
            """Get recent chat history."""
            if not self.services.mongo_handler or not self.services.mongo_handler.is_connected():
                return {"messages": []}
            
            bot_name = self.services.config_manager.get_bot_name()
            user_name = self.services.config_manager.get_user_name()
            
            raw_history = self.services.mongo_handler.get_recent_chat_history(
                count=limit,
                role_play_character=bot_name
            )
            
            messages = []
            for msg in raw_history:
                sender = msg.get("sender", "")
                
                # 处理时间戳 - 可能是datetime对象、float或字符串
                timestamp = msg.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, float):
                    # Unix时间戳
                    timestamp = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    # 已经是ISO格式字符串
                    timestamp_str = timestamp
                else:
                    # datetime对象
                    timestamp_str = timestamp.isoformat()
                
                if not isinstance(timestamp, str):
                    timestamp_str = timestamp.isoformat()
                
                messages.append(ChatMessage(
                    id=str(msg.get("_id", uuid.uuid4())),
                    sender=sender,
                    text=msg.get("message_text", ""),
                    timestamp=timestamp_str,
                    is_user=(sender == user_name),
                    emotion=msg.get("emotion")
                ))
            
            return {"messages": messages}
        
        @self.app.post("/api/chat/send", response_model=MessageResponse)
        async def send_message(request: MessageRequest):
            """Send a chat message."""
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
            
            # Save user message to database
            if self.services.mongo_handler and self.services.mongo_handler.is_connected():
                user_name = self.services.config_manager.get_user_name()
                bot_name = self.services.config_manager.get_bot_name()
                
                message_text = request.text
                if request.files:
                    message_text += f" [附加文件: {len(request.files)}个]"
                
                self.services.mongo_handler.insert_chat_message(
                    sender=user_name,
                    message_text=message_text,
                    role_play_character=bot_name
                )
            
            # Process message asynchronously
            asyncio.create_task(self._process_message(request, message_id))
            
            return MessageResponse(
                status="processing",
                message_id=message_id
            )
        
        @self.app.post("/api/upload")
        async def upload_file(file: UploadFile = File(...)):
            """Upload a file (image, audio, etc.)."""
            try:
                file_id = f"file_{uuid.uuid4().hex[:8]}"
                file_ext = os.path.splitext(file.filename)[1]
                filename = f"{file_id}{file_ext}"
                filepath = os.path.join(self.upload_dir, filename)
                
                # Save file
                with open(filepath, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                mime_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
                
                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "mime_type": mime_type,
                    "url": f"/api/files/{filename}"
                }
            except Exception as e:
                logger.error(f"Error uploading file: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/files/{filename}")
        async def get_file(filename: str):
            """Serve uploaded files."""
            filepath = os.path.join(self.upload_dir, filename)
            if not os.path.exists(filepath):
                raise HTTPException(status_code=404, detail="File not found")
            return FileResponse(filepath)
        
        @self.app.get("/api/avatar/{avatar_type}")
        async def get_avatar(avatar_type: str):
            """Get bot or user avatar."""
            if avatar_type == "bot":
                avatar_path = self.services.bot_avatar_path
            elif avatar_type == "user":
                avatar_path = self.services.user_avatar_path
            else:
                raise HTTPException(status_code=400, detail="Invalid avatar type")
            
            if not os.path.exists(avatar_path):
                raise HTTPException(status_code=404, detail="Avatar not found")
            
            return FileResponse(avatar_path)
        
        @self.app.post("/api/agent/toggle")
        async def toggle_agent_mode(request: AgentToggleRequest):
            """Toggle agent mode on/off."""
            if not self.services.agent_core:
                raise HTTPException(status_code=400, detail="Agent core not available")
            
            self.services.agent_core.set_agent_mode(request.enabled)
            
            # Broadcast to all connected clients
            await self.broadcast({
                "type": "agent_mode_changed",
                "enabled": request.enabled
            })
            
            return {"agent_mode": request.enabled}
        
        # Configuration Management  API Routes
        @self.app.get("/api/config/all")
        async def get_all_configs():
            """Get all configuration items organized by category."""
            try:
                all_configs = self.config_service.get_all_configs()
                # Convert ConfigItem objects to dicts for JSON response
                result = {}
                for category, items in all_configs.items():
                    result[category] = [item.model_dump() for item in items]
                return result
            except Exception as e:
                logger.error(f"Error getting all configs: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/config/category/{category}")
        async def get_category_configs(category: str):
            """Get configuration items for a specific category."""
            try:
                items = self.config_service.get_category_configs(category)
                return {"items": [item.model_dump() for item in items]}
            except Exception as e:
                logger.error(f"Error getting category configs: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/api/config")
        async def update_configs(request: ConfigUpdateRequest):
            """Update configuration values."""
            try:
                result = self.config_service.update_configs(request.updates)
                return result
            except Exception as e:
                logger.error(f"Error updating configs: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/config/save", response_model=ConfigSaveResponse)
        async def save_config():
            """Save configuration to INI file."""
            try:
                success, error_msg = self.config_service.save_to_file()
                if success:
                    return ConfigSaveResponse(
                        success=True,
                        message="配置已成功保存。请重启应用以使更改生效。"
                    )
                else:
                    return ConfigSaveResponse(
                        success=False,
                        message=error_msg or "保存配置失败"
                    )
            except Exception as e:
                logger.error(f"Error saving config: {e}", exc_info=True)
                return ConfigSaveResponse(
                    success=False,
                    message=f"保存配置时出错: {str(e)}"
                )
        
        @self.app.post("/api/system/restart")
        async def restart_application():
            """Restart the application."""
            try:
                logger.info("Application restart requested via API")
                
                # Broadcast to clients that restart is happening
                await self.broadcast({
                    "type": "system_restart",
                    "message": "应用程序正在重启..."
                })
                
                # Give time for the broadcast to send
                await asyncio.sleep(0.5)
                
                # Restart the application
                # This will use the same command line arguments
                python = sys.executable
                os.execl(python, python, *sys.argv)
                
                return {"success": True, "message": "应用程序正在重启..."}
            except Exception as e:
                logger.error(f"Error restarting application: {e}", exc_info=True)
                return {"success": False, "message": f"重启失败: {str(e)}"}
        
        # Preset Management API Routes
        @self.app.get("/api/presets")
        async def get_all_presets():
            """获取所有预设列表"""
            try:
                presets = self.preset_manager.get_all_presets()
                return {"presets": [preset.model_dump() for preset in presets]}
            except Exception as e:
                logger.error(f"Error getting presets: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/presets/{preset_id}")
        async def get_preset(preset_id: str):
            """获取单个预设详情"""
            try:
                preset = self.preset_manager.get_preset_by_id(preset_id)
                if preset is None:
                    raise HTTPException(status_code=404, detail=f"预设 {preset_id} 未找到")
                return preset.model_dump()
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting preset {preset_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/presets")
        async def create_preset(request: PresetCreateRequest):
            """创建新预设"""
            try:
                preset_data = request.model_dump()
                preset = self.preset_manager.create_preset(preset_data)
                if preset is None:
                    raise HTTPException(status_code=500, detail="创建预设失败")
                return {"success": True, "preset": preset.model_dump()}
            except Exception as e:
                logger.error(f"Error creating preset: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/api/presets/{preset_id}")
        async def update_preset(preset_id: str, request: PresetUpdateRequest):
            """更新预设"""
            try:
                # 只包含非None的字段
                update_data = {k: v for k, v in request.model_dump().items() if v is not None}
                preset = self.preset_manager.update_preset(preset_id, update_data)
                if preset is None:
                    raise HTTPException(status_code=404, detail=f"预设 {preset_id} 未找到")
                return {"success": True, "preset": preset.model_dump()}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating preset {preset_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/presets/{preset_id}")
        async def delete_preset(preset_id: str):
            """删除预设"""
            try:
                success = self.preset_manager.delete_preset(preset_id)
                if not success:
                    raise HTTPException(status_code=404, detail=f"预设 {preset_id} 未找到")
                return {"success": True, "message": "预设已删除"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting preset {preset_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/presets/{preset_id}/activate")
        async def activate_preset(preset_id: str):
            """激活预设并更新ConfigManager"""
            try:
                # 激活预设
                preset = self.preset_manager.activate_preset(preset_id)
                if preset is None:
                    raise HTTPException(status_code=404, detail=f"预设 {preset_id} 未找到")
                
                # 更新ConfigManager中的配置
                config_manager = self.services.config_manager
                config_manager.config.set("BOT", "NAME", preset.bot_name)
                config_manager.config.set("BOT", "PERSONA", preset.bot_persona)
                if preset.speech_pattern:
                    config_manager.config.set("BOT", "SPEECH_PATTERN", preset.speech_pattern)
                if preset.constraints:
                    config_manager.config.set("BOT", "CONSTRAINTS", preset.constraints)
                if preset.format_example:
                    config_manager.config.set("BOT", "FORMAT_EXAMPLE", preset.format_example)
                
                # 更新头像路径
                if preset.avatar_filename and self.preset_manager.avatar_exists(preset.avatar_filename):
                    avatar_path = self.preset_manager.get_avatar_path(preset.avatar_filename)
                    # 更新services中的bot_avatar_path
                    self.services.bot_avatar_path = avatar_path
                
                # 广播预设切换事件
                await self.broadcast({
                    "type": "preset_activated",
                    "preset_id": preset.id,
                    "preset_name": preset.name,
                    "bot_name": preset.bot_name
                })
                
                logger.info(f"预设已激活: {preset.name} (id={preset.id})")
                return {
                    "success": True,
                    "message": f"预设 '{preset.name}' 已激活",
                    "preset": preset.model_dump()
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error activating preset {preset_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/presets/from-current-config")
        async def create_preset_from_current_config():
            """从当前ConfigManager配置创建新预设"""
            try:
                config_manager = self.services.config_manager
                
                # 从当前配置读取信息
                bot_name = config_manager.get_bot_name()
                bot_persona = config_manager.get_bot_persona()
                speech_pattern = config_manager.get_bot_speech_pattern()
                constraints = config_manager.get_bot_constraints()
                format_example = config_manager.get_bot_format_example()
                
                # 生成预设名称
                preset_name = f"{bot_name}"
                
                # 创建预设数据
                preset_data = {
                    "name": preset_name,
                    "bot_name": bot_name,
                    "bot_persona": bot_persona,
                    "speech_pattern": speech_pattern,
                    "constraints": constraints,
                    "format_example": format_example,
                    "avatar_filename": "default_preset.png",
                    "is_active": False  # 新创建的预设默认不激活
                }
                
                # 创建预设
                preset = self.preset_manager.create_preset(preset_data)
                if preset is None:
                    raise HTTPException(status_code=500, detail="创建预设失败")
                
                logger.info(f"已从当前配置创建预设: {preset.name}")
                return {
                    "success": True,
                    "message": f"已保存当前配置为好友 '{preset.name}'",
                    "preset": preset.model_dump()
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error creating preset from current config: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/presets/avatars/upload")
        async def upload_preset_avatar(file: UploadFile = File(...)):
            """上传预设头像"""
            try:
                # 生成唯一文件名
                file_ext = os.path.splitext(file.filename)[1]
                filename = f"preset_{uuid.uuid4().hex[:8]}{file_ext}"
                filepath = self.preset_manager.get_avatar_path(filename)
                
                # 保存文件
                with open(filepath, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                logger.info(f"预设头像已上传: {filename}")
                return {
                    "success": True,
                    "filename": filename,
                    "url": f"/api/presets/avatars/{filename}"
                }
            except Exception as e:
                logger.error(f"Error uploading preset avatar: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/presets/avatars/{filename}")
        async def get_preset_avatar(filename: str):
            """获取预设头像"""
            try:
                filepath = self.preset_manager.get_avatar_path(filename)
                if not os.path.exists(filepath):
                    raise HTTPException(status_code=404, detail="头像文件未找到")
                return FileResponse(filepath)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting preset avatar {filename}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/presets/reorder")
        async def reorder_presets(request: Dict[str, List[str]]):
            """重新排序预设"""
            try:
                preset_ids = request.get("preset_ids", [])
                if not preset_ids:
                    raise HTTPException(status_code=400, detail="未提供预设ID列表")
                
                success = self.preset_manager.reorder_presets(preset_ids)
                if not success:
                    raise HTTPException(status_code=500, detail="重新排序失败")
                
                return {"success": True, "message": "预设已重新排序"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error reordering presets: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)
    
    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
        
        try:
            while True:
                data = await websocket.receive_json()
                
                # Handle ping/pong
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSocket clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
    
    async def _process_message(self, request: MessageRequest, message_id: str):
        """Process user message and get AI response."""
        try:
            # Notify clients that AI is typing
            await self.broadcast({
                "type": "typing",
                "is_typing": True
            })
            
            # Get AI response
            is_agent_mode = self.services.agent_core.is_agent_mode_active if self.services.agent_core else False
            
            if is_agent_mode and self.services.agent_core:
                # Agent mode
                response_data = await self.services.agent_core.process_user_request(
                    request.text,
                    media_files=[]  # File handling to be implemented
                )
            else:
                # Normal chat mode
                response_data = await self.services.gemini_client.send_message(
                    message_text=request.text,
                    hippocampus_manager=self.services.hippocampus_manager,
                    is_agent_mode=False
                )
            
            # Save AI response to database (only if not an error)
            if (self.services.mongo_handler 
                and self.services.mongo_handler.is_connected() 
                and not response_data.get("is_error", False)):
                bot_name = self.services.config_manager.get_bot_name()
                bot_text = response_data.get("text", "")
                
                self.services.mongo_handler.insert_chat_message(
                    sender=bot_name,
                    message_text=bot_text,
                    role_play_character=bot_name
                )
            
            # Stop typing indicator
            await self.broadcast({
                "type": "typing",
                "is_typing": False
            })
            
            # Broadcast AI response (only if not an error)
            if not response_data.get("is_error", False):
                bot_text = response_data.get("text", "")
                bot_emotion = response_data.get("emotion", "default")
                
                # Update desktop window立绘
                if self.services.bot_window and hasattr(self.services.bot_window, 'update_speech_and_emotion'):
                    try:
                        self.services.bot_window.update_speech_and_emotion(bot_text, bot_emotion)
                        logger.debug(f"桌面窗口已更新: emotion={bot_emotion}")
                    except Exception as e:
                        logger.warning(f"更新桌面窗口失败: {e}")
                
                await self.broadcast({
                    "type": "message",
                    "data": {
                        "message_id": f"msg_{uuid.uuid4().hex[:8]}",
                        "sender": self.services.config_manager.get_bot_name(),
                        "text": bot_text,
                        "emotion": bot_emotion,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
            else:
                # Log error but don't show to user
                logger.error(
                    f"API 调用失败，不向前端发送错误信息。错误内容: {response_data.get('text', '')}\n"
                    f"Thinking: {response_data.get('thinking_process', '')}"
                )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            
            await self.broadcast({
                "type": "typing",
                "is_typing": False
            })
            
            await self.broadcast({
                "type": "error",
                "message": f"处理消息时出错: {str(e)}"
            })


def create_app(services) -> FastAPI:
    """Create and configure the FastAPI application."""
    server = ChatWebServer(services)
    return server.app
