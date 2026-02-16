# keytao-bot

基于 NoneBot2 的跨平台聊天机器人，支持 QQ 和 Telegram 双平台。

## ✨ 功能特性

- 🤖 **跨平台支持**：同时支持 QQ 和 Telegram
- 💬 **智能回复**：自动回复用户消息并打招呼
- 🔧 **易于扩展**：基于插件系统，方便添加新功能
- 🚀 **现代化架构**：使用 uv 管理依赖，FastAPI 驱动

## 📦 项目结构

```
keytao-bot/
├── bot.py                 # 入口文件
├── pyproject.toml         # 项目配置和依赖
├── .env.dev              # 开发环境配置
├── .env.prod             # 生产环境配置
├── .env.example          # 配置模板
└── keytao_bot/
    ├── __init__.py
    └── plugins/          # 插件目录
        ├── __init__.py
        └── echo_hi.py    # 回复插件
```

## 🚀 快速开始

### 1. 安装依赖

项目使用 uv 管理依赖（已初始化）：

```bash
uv sync
```

### 2. 配置机器人

#### 配置 QQ 机器人

1. 访问 [QQ 开放平台](https://q.qq.com/) 注册机器人
2. 获取 Bot ID、Token 和 Secret
3. 在 `.env.dev` 或 `.env.prod` 中配置：

```bash
# 公域群机器人配置示例
QQ_BOTS='[{"id": "YOUR_BOT_ID", "token": "YOUR_BOT_TOKEN", "secret": "YOUR_BOT_SECRET", "intent": {"c2c_group_at_messages": true}, "use_websocket": false}]'
```

**Intent 配置说明**：
- `c2c_group_at_messages: true` - 群聊 @ 消息（群机器人必需）
- `guild_messages: true` - 频道消息（频道机器人必需）
- `at_messages: true` - @ 消息通知

#### 配置 Telegram 机器人

1. 在 Telegram 中找到 [@BotFather](https://t.me/botfather)
2. 发送 `/newbot` 创建机器人
3. 获得 Token 后配置到 `.env.dev` 或 `.env.prod`：

```bash
telegram_bots='[{"token": "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"}]'
```

**重要配置**：
- 向 @BotFather 发送 `/setprivacy` 并选择 `Disable`，让机器人能接收所有消息
- 如果在中国大陆，需要配置代理：`telegram_proxy="http://127.0.0.1:7890"`

#### 安装额外驱动（Telegram 需要）

Telegram 适配器需要 httpx 驱动器：

```bash
uv add httpx[http2]
```

或使用 nb-cli：

```bash
nb driver install httpx
```

### 3. 运行机器人

```bash
# 使用 nb-cli 运行（推荐）
nb run

# 或直接运行
python bot.py
```

开发时使用热重载：

```bash
nb run --reload
```

## 💡 使用说明

### 当前功能

机器人会响应 @ 或私聊消息，并回复：

```
用户发送：你好
机器人回复：你好 hi 用户名
```

### 测试机器人

**QQ 平台**：
1. 将机器人添加到群聊或频道
2. @ 机器人并发送消息：`@bot 你好`
3. 机器人会回复：`你好 hi [你的昵称]`

**Telegram 平台**：
1. 在 Telegram 中搜索你的机器人
2. 发送消息：`你好`
3. 机器人会回复：`你好 hi [你的用户名]`

## 🔧 开发指南

### 添加新插件

在 `keytao_bot/plugins/` 目录创建新的 `.py` 文件：

```python
from nonebot import on_command
from nonebot.adapters import Bot, Event

# 创建命令响应器
hello = on_command("hello", priority=10)

@hello.handle()
async def handle_hello(bot: Bot, event: Event):
    await hello.finish("Hello World!")
```

### 环境配置

- **开发环境**：使用 `.env.dev`，日志级别为 DEBUG
- **生产环境**：使用 `.env.prod`，日志级别为 INFO

切换环境：

```bash
# 方式 1：通过环境变量
ENVIRONMENT=dev nb run

# 方式 2：修改 .env.prod 中的 ENVIRONMENT 值
```

### 日志查看

日志会输出到控制台，可以通过 `LOG_LEVEL` 控制详细程度：
- `DEBUG`：最详细（开发推荐）
- `INFO`：一般信息（生产推荐）
- `WARNING`：警告信息
- `ERROR`：仅错误

## 📚 技术栈

- **NoneBot2**：机器人框架
- **adapter-qq**：QQ 官方接口适配器
- **adapter-telegram**：Telegram 适配器
- **FastAPI**：Web 框架（驱动器）
- **httpx**：HTTP 客户端（Telegram 需要）
- **uv**：依赖管理工具

## 🐛 常见问题

### QQ 机器人无法连接

1. 检查 Bot ID、Token、Secret 是否正确
2. 确认 Intent 配置是否匹配机器人类型（群机器人 vs 频道机器人）
3. 查看日志中的详细错误信息

### Telegram 机器人报 NetworkError

1. 检查网络连接
2. 如在中国大陆，确保配置了代理
3. 安装 httpx 驱动器：`uv add httpx[http2]`

### 机器人不响应消息

1. **QQ**：确认已 @ 机器人，或在私聊中发送
2. **Telegram**：确认向 @BotFather 发送了 `/setprivacy` 并选择 `Disable`
3. 检查插件中的 `rule=to_me()` 规则

### 如何让机器人响应所有消息（不需要 @）

修改 `keytao_bot/plugins/echo_hi.py`：

```python
# 移除 rule=to_me() 参数
echo_hi = on_message(priority=10, block=True)
```

⚠️ **注意**：这会让机器人响应所有消息，可能造成刷屏。

## 📖 参考文档

- [NoneBot2 官方文档](https://nonebot.dev/)
- [QQ 开放平台文档](https://bot.q.qq.com/wiki/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [adapter-qq 文档](https://github.com/nonebot/adapter-qq)
- [adapter-telegram 使用指南](https://github.com/nonebot/adapter-telegram/blob/beta/MANUAL.md)

## 📝 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

