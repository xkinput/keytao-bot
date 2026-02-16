# keytao-bot

基于 NoneBot2 的跨平台聊天机器人，支持 QQ 和 Telegram 双平台。

## ✨ 功能特性

- 🤖 **跨平台支持**：同时支持 QQ 和 Telegram
- 🧠 **AI 智能对话**：基于阿里云通义千问的智能聊天功能
- 🎯 **Skills 系统**：AI 自动识别需求并调用工具（Function Calling）
- 📖 **键道查词**：通过 skill 实现智能查词，AI 自动调用
- 🔧 **易于扩展**：基于插件和 skills 系统，方便添加新功能
- 🚀 **现代化架构**：使用 uv 管理依赖，FastAPI 驱动
- 🛡️ **合规设计**：内置中国法律法规和人道主义价值观约束

## 📦 项目结构

```
keytao-bot/
├── bot.py                 # 入口文件
├── pyproject.toml         # 项目配置和依赖
├── .env.dev              # 开发环境配置
├── .env.prod             # 生产环境配置
├── .env.example          # 配置模板
├── test_skills.py        # Skills 系统测试
└── keytao_bot/
    ├── __init__.py
    ├── plugins/          # 插件目录
    │   ├── __init__.py
    │   └── openai_chat.py    # DashScope (通义千问) AI 聊天插件
    └── skills/           # AI Skills (Function Calling)
        ├── __init__.py       # Skills Manager
        ├── README.md         # Skills 开发文档
        └── keytao-lookup/    # 键道查词 skill
            ├── SKILL.md      # Skill 说明
            └── tools.py      # 工具实现
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

#### 配置 Keytao 查词 API

查词功能使用 Keytao Next API，默认配置为：

```bash
KEYTAO_NEXT_URL="https://keytao.vercel.app"
```

如果你部署了自己的 Keytao Next 实例，可以修改此 URL。

#### 配置 DashScope API（AI 聊天功能）

1. 访问 [阿里云百炼平台](https://bailian.console.aliyun.com/) 获取 API Key
2. 在 `.env.dev` 或 `.env.prod` 中配置：

```bash
# DashScope API Key（必填）
DASHSCOPE_API_KEY="sk-your-api-key-here"

# 可选配置
DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  # API 地址
DASHSCOPE_MODEL="qwen-plus"                 # 使用的模型
DASHSCOPE_MAX_TOKENS=1000                  # 最大回复长度
DASHSCOPE_TEMPERATURE=0.7                  # 创造性 (0.0-2.0)
```

**模型选择建议**：
- `qwen-plus` - 推荐，性价比高，适合日常对话
- `qwen-turbo` - 更快，价格更低
- `qwen-max` - 最强能力，价格较高
- 更多模型：[模型列表](https://help.aliyun.com/model-studio/getting-started/models)

**兼容性说明**：
- DashScope 兼容 OpenAI 的 API 格式
- 可无缝切换到其他兼容 OpenAI 的服务商

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

#### 1. AI 智能对话（openai_chat）

机器人会使用阿里云通义千问进行智能对话，只需 @ 机器人或私聊即可：

```
你: @bot 你好，介绍一下键道输入法
机器人: 你好！键道输入法是一款基于双拼和字根的输入法...

你: @bot 帮我写一首关于春天的诗
机器人: [AI 生成的诗歌内容]
```

**特点**：
- ✅ 遵守中国法律法规
- ✅ 践行社会主义核心价值观
- ✅ 拒绝违法、暴力、色情等不良内容
- ✅ 保护隐私，不收集个人信息
- ✅ 内容客观、准确、有帮助

**优先级**：普通对话优先级最低（99）

#### 2. AI Skills（自动工具调用）

AI 会根据对话内容自动识别需求并调用相应工具（Function Calling）。

**示例 - 智能查词**：
```
你: @bot nau 这个编码对应什么词？
机器人: [自动调用 keytao_lookup_by_code]
      nau 对应的词是：你好

你: @bot 你好用键道怎么打？
机器人: [自动调用 keytao_lookup_by_word]
      "你好" 的键道编码是 nau
```

**特点**：
- 🎯 自动识别用户意图，无需输入固定命令
- 🔧 通过 Skills Manager 动态加载工具
- 📦 易于扩展，添加新 skill 无需修改 AI 代码
- 🔄 支持多轮对话和工具链式调用

**查看 Skills 开发文档**：[keytao_bot/skills/README.md](keytao_bot/skills/README.md)

### 测试 Skills 系统

运行测试脚本验证 skills 是否正常工作：

```bash
python3 test_skills.py
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
- **DashScope (通义千问)**：阿里云大模型 API，提供 AI 对话能力（兼容 OpenAI 格式）
- **adapter-qq**：QQ 官方接口适配器
- **adapter-telegram**：Telegram 适配器
- **FastAPI**：Web 框架（驱动器）
- **httpx**：HTTP 客户端（Telegram 和 API 请求）
- **uv**：现代化 Python 依赖管理工具

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

修改 [keytao_bot/plugins/openai_chat.py](keytao_bot/plugins/openai_chat.py)：

```python
# 移除 rule=to_me() 参数
ai_chat = on_message(priority=99, block=True)
```

⚠️ **注意**：这会让机器人响应所有消息，可能造成刷屏和大量 API 调用费用。

### DashScope API 调用失败

**错误提示**：`❌ AI 服务暂时不可用，请稍后重试`

**解决方法**：
1. 检查 `DASHSCOPE_API_KEY` 是否配置正确
2. 运行测试脚本：`python test_openai.py`
3. 查看详细日志：`LOG_LEVEL=DEBUG nb run`
4. 检查账户余额：[账户管理](https://bailian.console.aliyun.com/)

### 如何切换模型

在 `.env` 中修改：

```bash
# 使用 qwen-max（更强大但更贵）
DASHSCOPE_MODEL="qwen-max"

# 使用 qwen-turbo（更快更便宜）
DASHSCOPE_MODEL="qwen-turbo"

# 使用 qwen-plus（推荐，均衡）
DASHSCOPE_MODEL="qwen-plus"
```

完整模型列表：[阿里云百炼模型](https://help.aliyun.com/model-studio/getting-started/models)

## 📖 参考文档

### 框架和适配器
- [NoneBot2 官方文档](https://nonebot.dev/)
- [QQ 开放平台文档](https://bot.q.qq.com/wiki/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [adapter-qq 文档](https://github.com/nonebot/adapter-qq)
- [adapter-telegram 使用指南](https://github.com/nonebot/adapter-telegram/blob/beta/MANUAL.md)

### DashScope (阿里云通义千问)
- [阿里云百炼平台](https://bailian.console.aliyun.com/)
- [DashScope API 文档](https://help.aliyun.com/model-studio/developer-reference/api-details)
- [模型列表](https://help.aliyun.com/model-studio/getting-started/models)
- [错误代码](https://help.aliyun.com/model-studio/developer-reference/error-code)
- [定价说明](https://help.aliyun.com/model-studio/product-overview/billing)

### 项目文档
- [AI 聊天功能详细指南](docs/openai_chat_guide.md)
- [键道查词功能指南](docs/keytao_lookup_guide.md)

## 📝 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

