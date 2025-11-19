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
    pet_name: str
    user_name: str
    pet_avatar: str
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
        
        self._setup_cors()
        self._setup_routes()
        
        logger.info("ChatWebServer initialized")
    
    def _setup_cors(self):
        """Configure CORS to allow access from browsers (including mobile)."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins (for mobile access)
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS middleware configured")
    
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
                pet_name=self.services.config_manager.get_pet_name(),
                user_name=self.services.config_manager.get_user_name(),
                pet_avatar="/api/avatar/pet",
                user_avatar="/api/avatar/user",
                agent_mode_enabled=self.services.agent_core.is_agent_mode_active if self.services.agent_core else False,
                available_emotions=self.services.available_emotions
            )
        
        @self.app.get("/api/chat/history")
        async def get_chat_history(limit: int = 20):
            """Get recent chat history."""
            if not self.services.mongo_handler or not self.services.mongo_handler.is_connected():
                return {"messages": []}
            
            pet_name = self.services.config_manager.get_pet_name()
            user_name = self.services.config_manager.get_user_name()
            
            raw_history = self.services.mongo_handler.get_recent_chat_history(
                count=limit,
                role_play_character=pet_name
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
                pet_name = self.services.config_manager.get_pet_name()
                
                message_text = request.text
                if request.files:
                    message_text += f" [附加文件: {len(request.files)}个]"
                
                self.services.mongo_handler.insert_chat_message(
                    sender=user_name,
                    message_text=message_text,
                    role_play_character=pet_name
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
            """Get pet or user avatar."""
            if avatar_type == "pet":
                avatar_path = self.services.pet_avatar_path
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
                pet_name = self.services.config_manager.get_pet_name()
                pet_text = response_data.get("text", "")
                
                self.services.mongo_handler.insert_chat_message(
                    sender=pet_name,
                    message_text=pet_text,
                    role_play_character=pet_name
                )
            
            # Stop typing indicator
            await self.broadcast({
                "type": "typing",
                "is_typing": False
            })
            
            # Broadcast AI response (only if not an error)
            if not response_data.get("is_error", False):
                await self.broadcast({
                    "type": "message",
                    "data": {
                        "message_id": f"msg_{uuid.uuid4().hex[:8]}",
                        "sender": self.services.config_manager.get_pet_name(),
                        "text": response_data.get("text", ""),
                        "emotion": response_data.get("emotion", "default"),
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
