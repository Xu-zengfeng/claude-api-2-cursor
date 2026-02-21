# Claude API 2 Cursor

让 Cursor 用上 Claude Code 中转站 API。

## 思路

本项目作为中转代理，实现 Cursor 与 Claude API 之间的协议转换。核心流程如下：

```
┌─────────────┐
│   Cursor    │
│   客户端     │
└──────┬──────┘
       │ OpenAI 格式请求
       │ POST /v1/chat/completions
       ▼
┌─────────────────────────────────────────────────────────┐
│                    本代理服务                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  1. 接入鉴权 (ACCESS_API_KEY)                     │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  ▼                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  2. 请求格式转换                                   │  │
│  │     • messages 格式映射                           │  │
│  │     • system 提取为独立字段                       │  │
│  │     • tools 转换为 Anthropic 格式                 │  │
│  │     • Tool Use 智能修复（引号容错）               │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  ▼                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │  3. 转发到 Claude 中转站                          │  │
│  │     • 注入 API Key (x-api-key / Bearer)          │  │
│  │     • 设置 anthropic-version 头                   │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                       │
└──────────────────┼───────────────────────────────────────┘
                   │ Anthropic 格式请求
                   ▼
            ┌──────────────┐
            │ Claude 中转站 │
            │  (Relay API)  │
            └──────┬───────┘
                   │ 流式响应 (SSE)
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    本代理服务                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │  4. 响应格式转换                                   │  │
│  │     • content_block_delta → delta.content        │  │
│  │     • tool_use → function_call                   │  │
│  │     • 流式事件转 OpenAI SSE 格式                  │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                       │
└──────────────────┼───────────────────────────────────────┘
                   │ OpenAI 格式流式响应
                   ▼
            ┌─────────────┐
            │   Cursor    │
            │   客户端     │
            └─────────────┘
```

### 关键转换点

**请求转换 (OpenAI → Anthropic)**
- `messages[].role: "system"` → 提取为顶层 `system` 字段
- `tools[].function` → `tools[].input_schema`
- `tool_choice: "auto"` → `tool_choice: {"type": "auto"}`

**响应转换 (Anthropic → OpenAI)**
- `content_block_start` → 初始化 `choices[0].delta`
- `content_block_delta` → `delta.content` / `delta.function_call`
- `message_stop` → `[DONE]` 事件

**Tool Use 智能修复**
- 自动修复 JSON 中的引号错误
- 兼容 Cursor 扁平化的 `tool_uses` 格式
- 字段名映射容错（`tool_name` ↔ `name`）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```
PROXY_TARGET_URL=https://your-relay-station.com
PROXY_API_KEY=sk-xxx
PROXY_PORT=3029
ACCESS_API_KEY=your-access-key
```

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PROXY_TARGET_URL` | Claude 中转站地址 | `https://api.anthropic.com` |
| `PROXY_API_KEY` | 中转站 API Key | - |
| `PROXY_PORT` | 服务监听端口 | `3029` |
| `API_TIMEOUT` | 请求超时（秒） | `300` |
| `ACCESS_API_KEY` | 接入鉴权 Key（为空则不鉴权） | - |

### 3. 启动服务

```bash
python start.py
```

### 4. Cursor 配置

在 Cursor 设置中：
- **Base URL**: `http://[你的服务器 IP]:3029` 或 `http://[你的域名]/v1`
- **API Key**: 填写 `ACCESS_API_KEY` 的值
- **添加自定义模型**: 建议跟claude code官方模型一致，如 `claude-sonnet-4-5-20250929`

注意，Cursor 的 Base URL 不能是 `localhost`、`127.0.0.1` 等本地地址，需要是你的服务器 IP 或域名。如果想本地部署，可以结合内网穿透工具使用（如花生壳、ngrok、frp 等），将本地服务映射到公网地址。

## Docker 部署

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填写必要的配置项（参考上方环境变量表）。

### 2. 启动服务

```bash
docker compose up -d
```

### 3. 查看日志

```bash
docker compose logs -f
```

### 4. 停止服务

```bash
docker compose down
```

> 服务默认监听 `PROXY_PORT`（默认 3029），支持通过 `.env` 自定义端口。容器以非 root 用户运行，内置健康检查。

## API 路由

| 路由 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容接口（主路由） |
| `/v1/messages` | POST | Anthropic 原生格式透传 |
| `/health` | GET | 健康检查 |

## API Key 注入逻辑

服务会根据 `PROXY_API_KEY` 的前缀自动选择注入方式：
- `sk-` 开头 → `x-api-key` 请求头
- 其他 → `Authorization: Bearer` 请求头

## 踩坑点和建议
**cursor对模型名的玄学**
  与其说是玄学，实际上也是我还没搞清楚原理。表现的话，即使cursor的自定义api是配置在openai格式下，但你添加不同的模型名，cursor真可能发起不同格式的请求。所以按经验来说，模型名就用上面建议的 `claude-sonnet-4-5-20250929`，这个用起来目前倒没什么问题，别用opus的，我试了几次都不行，抓请求看，cursor发起的请求跟sonnet的不一样
