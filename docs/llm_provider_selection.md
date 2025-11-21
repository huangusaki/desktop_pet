# LLM 提供商选择配置说明

## 概述

现在可以通过配置文件选择使用 Gemini 或 OpenAI 作为主要的 LLM 提供商。

## 配置方法

在 `config/settings.ini` 中添加以下配置:

```ini
[API]
PRIMARY_LLM_PROVIDER = gemini  # 或 openai

[GEMINI]
API_KEY = your-gemini-api-key
MODEL_NAME = gemini-1.5-flash-latest
# ... 其他 Gemini 配置

[OPENAI]
ENABLED = True  # 是否启用 OpenAI
API_KEY = your-openai-api-key
MODEL_NAME = gpt-3.5-turbo
BASE_URL = https://api.openai.com/v1
TEMPERATURE = 0.7
MAX_TOKENS = 1000
TIMEOUT_SECONDS = 30
```

## 配置项说明

### PRIMARY_LLM_PROVIDER (必填)
- **类型**: string
- **可选值**: `gemini` 或 `openai`
- **默认值**: `gemini`
- **说明**: 选择主要使用的 LLM 提供商

### OpenAI 配置项

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ENABLED` | bool | `False` | 是否启用 OpenAI |
| `API_KEY` | string | `""` | OpenAI API 密钥 |
| `MODEL_NAME` | string | `"gpt-3.5-turbo"` | 模型名称 (如 gpt-4, gpt-3.5-turbo) |
| `BASE_URL` | string | `"https://api.openai.com/v1"` | API 基础 URL (可用于兼容的第三方 API) |
| `TEMPERATURE` | float | `0.7` | 温度参数 (0-2) |
| `MAX_TOKENS` | int | `1000` | 最大 token 数 |
| `TIMEOUT_SECONDS` | int | `30` | 请求超时时间 |

## 使用示例

### 使用 Gemini (默认)

```ini
[API]
PRIMARY_LLM_PROVIDER = gemini

[GEMINI]
API_KEY = AIzaSy...
MODEL_NAME = gemini-1.5-flash-latest
```

### 使用 OpenAI

```ini
[API]
PRIMARY_LLM_PROVIDER = openai

[OPENAI]
ENABLED = True
API_KEY = sk-...
MODEL_NAME = gpt-4
BASE_URL = https://api.openai.com/v1
TEMPERATURE = 0.7
MAX_TOKENS = 2000
```

### 使用兼容的第三方 API

```ini
[API]
PRIMARY_LLM_PROVIDER = openai

[OPENAI]
ENABLED = True
API_KEY = dummy_key
MODEL_NAME = llama-3-8b
BASE_URL = http://localhost:8000/v1  # 本地部署的模型
TEMPERATURE = 0.7
```

## 当前状态

✅ **已实现:**
- 配置 schema 定义
- ConfigManager 读取方法
- application_context.py 中的提供商选择逻辑
- 前端配置页面自动支持(通过 schema)

⚠️ **待完成 (TODO):**
- OpenAI 客户端到 GeminiClient 接口的适配器
- 完整的 OpenAI 聊天集成测试

**当前行为:**
- 如果选择 `openai` 作为主提供商,系统会尝试初始化 OpenAI 客户端
- 由于适配器尚未实现,系统会自动回退到 Gemini
- 会在日志中显示警告信息

## 前端配置

在 Web UI 的配置页面中,**API配置** 类别会自动显示所有配置项,包括:
- 主要LLM提供商选择
- Gemini 配置
- OpenAI 配置

用户可以直接在 Web UI 中修改这些配置。

## 注意事项

1. **API Key 安全**: API 密钥会在配置页面中以密码形式显示
2. **配置验证**: 修改配置后需要保存并重启应用才能生效
3. **回退机制**: 如果选择的提供商初始化失败,系统会自动回退到 Gemini
4. **日志监控**: 查看应用日志了解当前使用的提供商和初始化状态
