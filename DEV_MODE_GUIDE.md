# Arisu 桌面宠物 - 开发指南

## 快速开始

### 方式一：直接运行（推荐）

```bash
# 安装依赖
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 构建前端
cd frontend && npm run build && cd ..

# 运行应用
python -m src.main
```

应用启动后：
- Web服务器自动运行在 http://localhost:8765
- 双击桌面宠物打开聊天窗口
- 或直接在浏览器访问 http://localhost:8765

### 方式二：开发模式（前端热重载）

**终端1 - 后端**：
```bash
python -m uvicorn src.web.server:create_app --factory --reload --host 0.0.0.0 --port 8765
```

**终端2 - 前端**：
```bash
cd frontend
npm run dev
```

访问：
- **开发服务器**: http://localhost:5173 （前端热重载）
- **生产服务器**: http://localhost:8765 （需先构建）
- **API文档**: http://localhost:8765/docs

## 功能特性

- ✅ 桌面宠物（PyQt6）
- ✅ Web聊天界面（React + Vite + TailwindCSS）
- ✅ 实时消息（WebSocket）
- ✅ 文件上传
- ✅ 移动端访问
- ✅ 屏幕截图分析
- ✅ Agent模式

## 常见问题

### 端口被占用
修改 `src/utils/application_context.py` 中的端口号（默认8765）

### 前端修改后不生效
运行 `cd frontend && npm run build` 重新构建

### WebSocket连接失败
检查防火墙是否允许8765端口
