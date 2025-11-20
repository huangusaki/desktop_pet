# 🎀 Arisu (爱丽丝) - 智能桌面宠物

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Latest-47A248?style=flat-square&logo=mongodb&logoColor=white)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-8E75B2?style=flat-square)

> **Arisu** 是一个结合了现代 Web 技术与桌面应用能力的智能助手。她不仅是你桌面上的可爱伙伴，更是拥有长期记忆与视觉感知能力的 AI 伴侣。

---

## ✨ 功能特性 (Features)

*   🤖 **深度智能交互**
    *   集成 **Google Gemini** 大语言模型，提供流畅、自然的对话体验。
    *   支持角色扮演 (Roleplay) 与情感反馈系统。

*   🧠 **海马体记忆系统 (Hippocampus Memory)**
    *   基于 **NetworkX** 构建的图导向记忆网络。
    *   能够建立概念关联，记住用户的喜好、过往对话及重要信息。

*   👁️ **视觉感知 (Vision Capabilities)**
    *   具备屏幕截图分析能力，能够“看”懂你当前的工作或娱乐内容并给出建议。

*   💻 **现代化技术栈**
    *   **前端**: 使用 **React 19** + **TypeScript** + **TailwindCSS** 构建，界面精美流畅。
    *   **后端**: 基于 **FastAPI** 的高性能异步服务。
    *   **桌面端**: **PyQt6** 驱动的桌面悬浮窗，支持透明背景与鼠标穿透。

*   📱 **多端无缝体验**
    *   内置 Web 服务器，支持局域网内通过手机或浏览器远程访问聊天界面。

---

## 🛠️ 技术栈 (Tech Stack)

| 领域 | 技术/库 | 说明 |
| :--- | :--- | :--- |
| **Core** | Python 3.x, TypeScript | 核心开发语言 |
| **Desktop** | PyQt6, PyQt6-WebEngine | 桌面应用框架与浏览器内核嵌入 |
| **Backend** | FastAPI, Uvicorn, Aiohttp | 高性能异步 Web 框架 |
| **Frontend** | React 19, Vite, TailwindCSS | 现代化前端开发体验 |
| **Database** | MongoDB (PyMongo) | 聊天记录与非结构化数据存储 |
| **AI / LLM** | Google Gemini API | 核心智能驱动 |
| **Memory** | NetworkX | 图结构记忆网络实现 |

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

确保你的系统已安装：
*   Python 3.10+
*   Node.js 18+
*   MongoDB (本地或远程服务)

### 2. 安装依赖

```bash
# 克隆项目 (如果你还没有)
# git clone https://github.com/your-username/Arisu_gf.git
# cd Arisu_gf

# 1. 安装 Python 后端依赖
pip install -r requirements.txt

# 2. 安装前端依赖
cd frontend
npm install
cd ..
```

### 3. 配置文件

在 `config` 目录下，确保配置了必要的 API Key 和数据库连接信息。
*   检查 `config/settings.ini` (如果不存在，请参考 `settings.ini.backup` 创建)。
*   确保填入有效的 **Google Gemini API Key**。

### 4. 运行应用

**推荐方式 (一键启动)**：

```bash
# 这一步会先构建前端，然后启动后端服务
cd frontend && npm run build && cd ..
python -m src.main
```

启动成功后：
*   桌面右下角会出现 Arisu 的悬浮窗。
*   Web 界面默认运行在: `http://localhost:8765`

---

## 💻 开发模式 (Development)

如果你需要修改前端代码并希望实时预览：

**终端 1 (后端服务)**:
```bash
python -m uvicorn src.web.server:create_app --factory --reload --host 0.0.0.0 --port 8765
```

**终端 2 (前端热重载)**:
```bash
cd frontend
npm run dev
```

*   **开发预览**: `http://localhost:5173`
*   **API 文档**: `http://localhost:8765/docs`

---

## 📂 目录结构

```
Arisu_gf/
├── src/                # Python 后端源码
│   ├── database/       # MongoDB 数据库操作
│   ├── memory_system/  # 记忆系统核心逻辑
│   ├── web/            # FastAPI Web 服务
│   └── ...
├── frontend/           # React 前端源码
├── config/             # 配置文件
├── scripts/            # 辅助脚本
└── requirements.txt    # Python 依赖列表
```

---

## 📄 License

[MIT License](LICENSE)