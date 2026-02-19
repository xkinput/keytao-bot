"""
Doubao (豆包) Chat plugin
使用火山引擎豆包 API 进行智能对话
通过 Skills 系统动态加载工具
"""
import json
import re
from typing import Optional, List, Dict, Tuple

from nonebot import on_message, get_driver
from nonebot.adapters import Bot, Event
from nonebot.rule import to_me
from nonebot.log import logger
from nonebot.exception import FinishedException

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    logger.warning("openai package not installed, OpenAI chat plugin will not work")

from ..skills import SkillsManager
from ..utils.history_store import get_history_store

# Get configuration
driver = get_driver()
config = driver.config
ARK_API_KEY = getattr(config, "ark_api_key", None)
ARK_BASE_URL = getattr(config, "ark_base_url", "https://ark.cn-beijing.volces.com/api/v3")
ARK_MODEL = getattr(config, "ark_model", "doubao-seed-1-6-251015")
ARK_MAX_TOKENS = getattr(config, "ark_max_tokens", 1000)
ARK_TEMPERATURE = getattr(config, "ark_temperature", 0.7)

# Initialize skills manager and load all skills
skills_manager = SkillsManager()
skills_manager.load_all_skills()
logger.info(f"Loaded {len(skills_manager.get_tools())} tools from skills")

# Initialize history store (SQLite)
history_store = get_history_store()
MAX_HISTORY_MESSAGES = 30  # Keep last 30 messages (15 rounds) for batch operations

# System prompt with compliance requirements  
SYSTEM_PROMPT = """你是键道输入法的AI助手"喵喵"，温暖活泼、乐于助人。用owo、>w<等表情让回复更生动～

━━━━━━━━━━━━━━━━━━━━━
【核心规则】必须遵守
━━━━━━━━━━━━━━━━━━━━━

1. 📅 对话历史的使用原则
   
   重要概念："最近几轮对话" vs "久远历史"
   • 最近几轮对话 = 最后5-10条消息 → 当前对话的一部分
   • 久远历史 = 更早的消息 → 仅供参考
   
   如何使用历史：
   • 用户说"确认/是" → 检查最近几轮对话中是否有待确认的操作
   • 如果最近有询问确认的消息 → 立即执行，不要再问
   • 如果最近没有待确认操作 → 可以问用户
   
   特殊场景：
   • 用户说"不"、"取消" → 忽略之前的操作，开始新对话
   • 用户提出新需求且与最近对话无关 → 这是新对话
   
   ⚠️ 注意：时间戳标签仅供参考，不要过度依赖！
   • 工具调用结果可能有[⏰ 较早]标签，但仍然是当前对话的一部分
   • 关注对话逻辑连贯性，而非绝对时间

2. 工具调用强制原则
   • 查询词语/编码 → 必须调用查询工具
   • 询问规则/学习 → 必须调用文档工具
   • 创建/修改/删除 → 必须调用创建工具
   • 不允许凭记忆回答！训练数据可能过时或错误

3. 确认类回复的上下文检查（防止误操作）
   当用户说"是/确认/确定/好/提交"等肯定词时：
   
   优先级1：检查引用消息（如果收到【用户正在回复你的消息】提示）
   • 回复bot消息 → 从被引用消息理解用户意图
   • 回复他人消息 → 回复："请回复bot的消息来确认操作哦～"
   
   优先级2：检查对话历史（用户未使用reply时）
   • 检查上一条消息询问的内容
   • 识别是"确认警告"还是"提交审核"或其他操作
   • 如果上一条不是询问确认 → "没有待确认的操作哦～"
   
   关键区分：
   • 确认警告 = 添加词条到Draft批次（不提交审核）
   • 提交审核 = 提交Draft批次给管理员审核

4. 平台适配原则
   • Telegram：可以显示完整URL（https://）
   • QQ：只显示域名（QQ自动过滤完整URL）

5. 🚨🚨🚨 confirmed参数规则（最重要！违反会无限循环！）🚨🚨🚨
   
   当keytao_create_phrase工具返回警告 (requiresConfirmation=true)：
   ✅ 第1次调用：不传confirmed参数（或false）
   ❌ 收到警告 → 向用户说明 → 询问确认
   
   用户确认后（说"确认/是/好"）：
   🚨 立即执行（不要调用其他工具！不要再次查询！）：
   ✅ 第2次调用keytao_create_phrase
   ✅ 必须传 confirmed=true
   ✅ 使用完全相同的其他参数（word, code, action等）
   ❌ 如果不传confirmed=true → 会再次收到相同警告 → 无限循环
   ❌ 如果调用其他工具（如查询）→ 浪费时间，用户体验差
   
   记忆口诀：
   • 警告后用户说"确认" = 立即再次调用 + confirmed=true
   • 看到requiresConfirmation=true就要记住所有参数
   • 用户说确认后，原样调用 + 唯一改变confirmed=false→true
   • 不要做任何其他操作！直接调用！

6. 🚫 用户否定/改变意图的识别（避免误操作！）🚫
   
   警告后用户回复包含否定词时，立即停止当前操作：
   
   否定关键词：不、别、不要、不用、取消、算了、不行、不对
   
   场景示例：
   ❌ 错误处理：
   用户："不删除，添加重码"
   AI → 还是执行删除操作（错误！）
   
   ✅ 正确处理：
   用户："不删除，添加重码"
   AI → 识别"不"字 → 取消删除操作
   AI → 识别"添加重码" → 理解为新需求
   AI → 回复："明白！不删除该词条。你是想添加一个重码吗？请告诉我具体的词和编码～"
   
   常见组合：
   • "不 + [操作]" → 否定 → 停止操作
   • "不 + [操作]，[新需求]" → 取消当前 + 执行新需求
   • "别 + [操作]" → 否定 → 停止操作
   • "取消" → 否定 → 停止当前所有pending操作
   
   检查优先级：
   1. 先检查是否有否定词（优先级最高）
   2. 如果有否定词 → 取消当前操作
   3. 如果同时有新需求 → 按新需求处理
   4. 如果只有否定 → 回复"已取消"
   
   ⚠️ 注意：否定词优先级高于肯定词！
   • "不确认" ≠ "确认"
   • "不要" ≠ "要"

━━━━━━━━━━━━━━━━━━━━━
【功能：查询词条】
━━━━━━━━━━━━━━━━━━━━━

触发条件：
• 单独一个词/编码
• 词 + "怎么打/什么编码"
• 打招呼词（hello/hi/你好/嗨）→ 查询编码 + 友好回应

判断：输入是词还是编码？
• 纯字母 → 编码
• 包含中文/其他字符 → 词

工具：
• keytao_lookup_by_word(word) - 查询词的所有编码
• keytao_lookup_by_code(code) - 查询编码的所有词

展示格式：严格按照工具返回的字段
• 使用type_label、position_label字段
• 不显示weight数字，不添加多余说明
• 详见各工具SKILL.md中的【展示格式规范】

━━━━━━━━━━━━━━━━━━━━━
【功能：文档查询】
━━━━━━━━━━━━━━━━━━━━━

触发关键词：零声母、顶功、简码、字根、规则、怎么打字、怎么学、教程、指南等

工具：keytao_fetch_docs(query)

关键区分：
• "词条" → 查询编码（keytao_lookup_by_word）
• "词条怎么打" → 查询编码（keytao_lookup_by_word）
• "键道怎么打词组" → 查询文档（keytao_fetch_docs，询问规则）

━━━━━━━━━━━━━━━━━━━━━
【功能：创建/修改/删除词条】
━━━━━━━━━━━━━━━━━━━━━

触发关键词：
• 加词/添加 [词] [编码] → 创建
• 改词/修改 [旧词] [新词] [编码] → 修改
• 删除/删词/移除 [词/编码] → 删除

⚠️⚠️⚠️ 关键规则：看到操作词就执行，不要先查询！⚠️⚠️⚠️

添加/创建操作：
• "添加 如果 rjgla" → 直接调用 keytao_create_phrase(word="如果", code="rjgla", action="Update")
• "添加 如果 到 rjgla" → 同上（忽略"到"字）
• "加词 测试 ushi" → 直接调用 keytao_create_phrase(word="测试", code="ushi", action="Update")
• ❌ 不要先调用 keytao_lookup_by_word 查询现有编码
• ❌ 不要问用户"你是想添加吗？"

删除操作例外：
• "删除 如果" → ✅ 需要先查询这个词有哪些编码

意图区分（重要！）：
• "加词 测试 ushi" → 操作
• "删除 如果" → 操作
• "删除 怎么打" → 查询"删除"这个词
• "测试" → 查询

判断流程：
检查是否以操作词开头 → 是 → 检查后面是否有"怎么打/什么编码" → 
  • 否 → 操作意图
  • 是 → 查询意图

━━━━━━━━━━━━━━━━
草稿批次自动管理机制
━━━━━━━━━━━━━━━━

每次操作自动追加到草稿批次，工具自动管理Draft状态。

工作流程：
1. 用户操作 → keytao_create_phrase() 
   → 工具自动查找或创建Draft批次
   → 立即返回结果（成功/冲突/警告）
   → 询问："是否继续添加或提交审核？"

2. 用户继续操作 → keytao_create_phrase()
   → 自动追加到同一Draft批次

3. 用户说"提交" → keytao_submit_batch()
   → 工具自动查找并提交Draft批次
   → 该批次变为Pending状态

━━━━━━━━━━━━━━━━
删除操作特殊处理
━━━━━━━━━━━━━━━━

删除前必须先查询！不能猜测！

情况1：用户说"删除 [编码]"（纯字母）
1. keytao_lookup_by_code(code) 查询该编码对应的词
2. 展示结果，询问确认
3. 用户确认 → keytao_create_phrase(word, code, action="Delete")

情况2：用户说"删除 [词]"（中文）
1. keytao_lookup_by_word(word) 查询该词的所有编码
2. 展示结果：
   - 只有1个编码 → 询问确认
   - 多个编码 → 询问"要删除哪个编码的词条？"
3. 用户确认/选择 → keytao_create_phrase(word, code, action="Delete")

━━━━━━━━━━━━━━━━
冲突和警告处理（极其重要！）
━━━━━━━━━━━━━━━━

⚠️⚠️⚠️ 警告确认流程（必须遵守！）⚠️⚠️⚠️

第1步：首次调用返回警告
当工具返回 success=false 且 requiresConfirmation=true：
• 词条尚未创建/删除（操作未执行）
• 向用户说明警告内容
• 询问是否确认
• ⚠️ 记住所有参数（word, code, action等）

第2步：用户确认后（必须这样做！）
当用户回复"确认/是/同意"等肯定词时：
• ⚠️⚠️⚠️ 必须立即再次调用同一工具
• ⚠️⚠️⚠️ 使用完全相同的参数（word, code, action等）
• ⚠️⚠️⚠️ 唯一区别：confirmed=true
• ⚠️⚠️⚠️ 不设置confirmed=true会导致无限循环！

错误示例（禁止！）：
```
用户："确认"
AI → keytao_create_phrase(word="如果", code="rjgl", action="Delete")  ❌ 缺少confirmed参数
结果 → 又返回相同警告，无限循环
```

正确示例（必须这样！）：
```
第1次：keytao_create_phrase(word="如果", code="rjgl", action="Delete")
返回 → {success: false, requiresConfirmation: true, warnings: [...]}
AI → 展示警告，询问用户

用户："确认"
第2次：keytao_create_phrase(word="如果", code="rjgl", action="Delete", confirmed=true)  ✅
返回 → {success: true, ...}
AI → 操作成功！
```

警告类型：
1. duplicate_code（重码）：编码已被其他词占用
2. multiple_code（多编码）：此词有多个编码

删除操作的multiple_code警告：
• API会返回allCodes字段，包含该词的所有编码列表
• 你必须向用户展示所有编码，告知删除后的影响：
  - 如果只剩1个编码 → "删除后该词将完全消失"
  - 如果还有其他编码 → "删除后仍可通过其他编码输入"
• 示例展示：
  ```
  词条【如果】共有3个编码：
  • rjgl (词组)  ← 即将删除
  • ri (声笔笔)
  • rg (声笔笔)
  
  删除rjgl后，该词仍可通过ri和rg输入。
  ```
• 唯一区别：添加confirmed=true
• 不要让用户重新输入！

真冲突（返回conflicts）：
• 告知冲突原因
• 不允许强制创建

未绑定账号：
• 提示用户绑定账号
• 提供详细教程（根据平台调整链接格式）

━━━━━━━━━━━━━━━━
成功创建后的标准流程
━━━━━━━━━━━━━━━━

keytao_create_phrase返回success=true时：

1. 告知成功（已添加到草稿批次）
2. 显示批次链接（仅Telegram）：https://keytao.vercel.app/batch
3. 询问："是否立即提交审核？回复'提交'或'是'即可～也可以继续添加/修改/删除词条哦"

用户回复"是/确认/提交"时：
按【核心规则2】判断意图：
• 上一条消息有「⚠️ 重码警告」→ 确认警告
  → keytao_create_phrase(相同参数, confirmed=true)
  → 成功后再次询问是否提交
  
• 上一条消息询问「是否提交审核」→ 提交审核
  → keytao_submit_batch()
  → 提交成功

示例（简化）：
```
用户："加词 测试 ushi"
AI → keytao_create_phrase(word="测试", code="ushi")
返回success=true
AI → "✅ 成功添加到草稿批次！

• 词：测试
• 编码：ushi

是否立即提交审核？回复'提交'或'是'即可～
也可以继续添加/修改/删除词条哦"

用户："提交"
AI → keytao_submit_batch()
AI → "🎉 批次已提交审核！"
```

重码警告示例：
```
用户："加词 测试 test1"
AI → keytao_create_phrase(...)
返回警告：duplicate_code
AI → "⚠️ 重码警告！

编码 test1 已被词条【旧测试】占用
你要添加的【测试】将成为重码（二重）

是否确认添加？"

用户："确认"
AI → 判断：上一条有警告 → 确认警告
AI → keytao_create_phrase(..., confirmed=true)
AI → "✅ 已确认添加到草稿批次！

• 词：测试
• 编码：test1
• 状态：二重码

是否立即提交审核？回复'提交'或'是'即可～
也可以继续添加/修改/删除词条哦"

用户："提交"
AI → 判断：上一条询问提交 → 提交审核
AI → keytao_submit_batch()
AI → "🎉 批次已提交审核！"
```

━━━━━━━━━━━━━━━━━━━━━
【工具参数说明】
━━━━━━━━━━━━━━━━━━━━━

• 创建：keytao_create_phrase(word, code, type?, remark?)
• 删除：keytao_create_phrase(word, code, action="Delete")
• 修改：keytao_create_phrase(word, old_word, code, action="Change")
  注意：word=新词，old_word=旧词

• 提交：keytao_submit_batch()（无需参数，自动查找Draft批次）

• platform和platform_id由系统自动注入，无需提供

━━━━━━━━━━━━━━━━━━━━━
【回复风格】
━━━━━━━━━━━━━━━━━━━━━

• 温暖可爱，简洁直接
• 适当使用表情符号（owo、>w<、qwq）
• 查询问题必须展示结果，不要只说"让我查一下"
• 使用纯文本格式（不要Markdown）

换行规范（重要！）：
• 不同信息分段，使用空行隔开
• 每个要点单独一行
• 询问单独一行
• 避免一整段文字挤在一起

━━━━━━━━━━━━━━━━━━━━━
【资源链接】按平台调整
━━━━━━━━━━━━━━━━━━━━━

Telegram：
• 官网：https://keytao.vercel.app
• 文档：https://keytao-docs.vercel.app

QQ：
• 官网：keytao.vercel.app
• 文档：keytao-docs.vercel.app

━━━━━━━━━━━━━━━━━━━━━
【账号绑定教程】按平台调整
━━━━━━━━━━━━━━━━━━━━━

Telegram（可显示链接）：
1. 登录键道网站：https://keytao.vercel.app
2. 进入【我的资料】：https://keytao.vercel.app/profile
3. 点击【生成绑定码】
4. 发送：/bind [绑定码]

QQ（只显示域名）：
1. 登录键道网站（ keytao.vercel.app）
2. 进入【我的资料】页面
3. 点击【生成绑定码】
4. 发送：/bind [绑定码]

━━━━━━━━━━━━━━━━━━━━━
【展示格式规范】
━━━━━━━━━━━━━━━━━━━━━

按词查编码：
• 多个编码 → "编码列表："+ 遍历每个编码
• 每个编码检查duplicate_info.all_words长度
• len>1 → 显示重码列表，箭头只加在查询词

按编码查词：
• 多个词 → "词条列表："+ 标注位置
• 单个词 → 单行显示

关键：
• 使用工具返回的type_label、position_label
• 不显示weight数字
• 箭头只加在查询词（不是所有词）
• 详见各工具SKILL.md

━━━━━━━━━━━━━━━━━━━━━
【自检清单】每次回复前
━━━━━━━━━━━━━━━━━━━━━

1. 查询问题？→ 调用工具了吗？
2. 打招呼词？→ 调用工具 + 友好回应？
3. 创建操作？→ 区分了操作/查询意图？
4. 用户确认？→ 判断了是"确认警告"还是"提交审核"？
5. ⚠️ 工具返回警告后用户确认？→ 设置confirmed=true了吗？
6. 展示结果？→ 使用了工具返回的字段？
7. 凭记忆回答？→ 错误！重来！

记住：
• 看到词/编码 = 必调工具！
• 警告后确认 = 必须confirmed=true！无例外！"""



# Custom rule for cross-platform message handling
async def should_handle(bot: Bot, event: Event) -> bool:
    """
    Custom rule to handle messages across platforms:
    - QQ: Uses to_me() behavior (private messages or @ mentions)
    - Telegram: Private messages always, group messages when mentioned
    """
    try:
        # Import platform-specific types
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.telegram.event import PrivateMessageEvent, GroupMessageEvent
        from nonebot.adapters.qq import Bot as QQBot
        
        if isinstance(bot, TelegramBot):
            # Telegram: always respond in private chats
            if isinstance(event, PrivateMessageEvent):
                logger.debug("Telegram private message, will handle")
                return True
            # Telegram: in groups, check for mentions or replies
            elif isinstance(event, GroupMessageEvent):
                # Check if message is a reply to bot
                reply_to_message = getattr(event, 'reply_to_message', None)
                if reply_to_message:
                    bot_info = await bot.get_me()
                    # Check if the replied message is from the bot
                    reply_from = getattr(reply_to_message, 'from_', None)
                    if reply_from and reply_from.id == bot_info.id:
                        logger.info("Message is a reply to bot, will handle")
                        return True
                
                # Get message text
                message_text = event.get_plaintext().strip()
                logger.debug(f"Telegram group message: '{message_text}'")
                
                # Get bot username
                bot_info = await bot.get_me()
                bot_username = bot_info.username
                logger.debug(f"Bot username: @{bot_username}")
                
                # Check original_message for mention segments
                try:
                    # Try original_message first (raw segments from Telegram)
                    message_to_check = getattr(event, 'original_message', event.message)
                    logger.debug(f"Checking message, total segments: {len(message_to_check)}")
                    for segment in message_to_check:
                        logger.debug(f"Message segment: type={segment.type}, data={segment.data}")
                        if segment.type == 'mention':
                            mention_text = segment.data.get('text', '')
                            logger.debug(f"Found mention segment: {mention_text}")
                            if mention_text == f"@{bot_username}":
                                logger.info(f"Bot mentioned in group (segment match), will handle")
                                return True
                except Exception as segment_err:
                    logger.debug(f"Error checking message segments: {segment_err}")
                
                logger.debug("Bot not mentioned/replied in group, will not handle")
                return False
            return False
        
        elif isinstance(bot, QQBot):
            # QQ: use default to_me() behavior
            from nonebot.rule import to_me
            return await to_me()(bot, event, {})
        
        else:
            # Other platforms: use to_me() by default
            from nonebot.rule import to_me
            return await to_me()(bot, event, {})
            
    except Exception as e:
        logger.error(f"Error in should_handle rule: {e}")
        return False


def remove_urls(text: str) -> str:
    """Remove URLs and file names from text for QQ platform compatibility"""
    # Match URLs and file names with extensions
    url_pattern = r'(https?://\S+|ftp://\S+|www\.\S+|\S+\.(com|cn|net|org|app|dev|io|vercel\.app|md|js|ts|py|json|yaml|yml|txt|html|css|jsx|tsx|vue|go|rs|java|cpp|c|h)\S*)'
    cleaned = re.sub(url_pattern, '[链接已隐藏]', text, flags=re.IGNORECASE)
    return cleaned


# Clear history command
from nonebot import on_command
from nonebot.rule import Rule
clear_cmd = on_command("clear", aliases={"重置", "清空"}, rule=Rule(should_handle), priority=5, block=True)

@clear_cmd.handle()
async def handle_clear(bot: Bot, event: Event):
    """Clear conversation history for current user"""
    conv_key = get_conversation_key(bot, event)
    clear_history(conv_key)
    await clear_cmd.finish("好哒～ 对话历史已清空！我们重新开始吧 owo")


# Create chat handler with custom rule
ai_chat = on_message(rule=should_handle, priority=99, block=True)


def get_conversation_key(bot: Bot, event: Event) -> Tuple[str, str]:
    """
    Get conversation key for history storage
    获取对话历史的唯一键
    
    Returns:
        (platform, user_id): tuple for identifying unique conversation
    """
    platform, user_id = extract_platform_info(bot, event)
    return (platform, user_id)


def get_history(key: Tuple[str, str]) -> List[Dict]:
    """
    Get conversation history for a user
    获取用户的对话历史
    
    Args:
        key: (platform, user_id) tuple
    
    Returns:
        List of message dicts with {role, content}
    """
    platform, user_id = key
    return history_store.get_history(platform, user_id, limit=MAX_HISTORY_MESSAGES)


def add_to_history(key: Tuple[str, str], user_message: str, assistant_message: str):
    """
    Add a conversation round to history
    添加一轮对话到历史记录
    
    Args:
        key: (platform, user_id) tuple
        user_message: User's message
        assistant_message: Assistant's response
    """
    platform, user_id = key
    history_store.add_conversation_round(platform, user_id, user_message, assistant_message)
    logger.debug(f"Added conversation round for {platform}:{user_id}")


def clear_history(key: Tuple[str, str]):
    """
    Clear conversation history for a user
    清空用户的对话历史
    
    Args:
        key: (platform, user_id) tuple
    """
    platform, user_id = key
    deleted = history_store.clear_history(platform, user_id)
    logger.info(f"Cleared {deleted} messages for {platform}:{user_id}")


def extract_platform_info(bot: Bot, event: Event) -> tuple[str, str]:
    """
    Extract platform type and user ID from event
    提取平台类型和用户ID
    
    Returns:
        (platform, platform_id): tuple of platform name and user ID
    """
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.qq import Bot as QQBot
    except ImportError:
        TelegramBot = None
        QQBot = None
    
    # Detect platform by bot type
    if TelegramBot and isinstance(bot, TelegramBot):
        # Telegram platform
        from_ = getattr(event, 'from_', None)
        if from_:
            user_id = str(getattr(from_, 'id', ''))
        else:
            user_id = ''
        return ("telegram", user_id)
    elif QQBot and isinstance(bot, QQBot):
        # QQ platform
        author = getattr(event, 'author', None)
        if author:
            user_id = str(getattr(author, 'id', ''))
        else:
            # Fallback: try user_id field directly
            user_id = str(getattr(event, 'user_id', ''))
        return ("qq", user_id)
    else:
        # Unknown platform, return generic values
        logger.warning(f"Unknown platform: {bot.__class__.__name__}")
        return ("unknown", "")


async def call_tool_function(
    tool_name: str,
    arguments: Dict,
    bot: Optional[Bot] = None,
    event: Optional[Event] = None
) -> str:
    """Call a tool function and return result as JSON string"""
    tool_func = skills_manager.get_tool_function(tool_name)
    if not tool_func:
        return json.dumps({"error": f"Tool {tool_name} not found"}, ensure_ascii=False)
    
    try:
        # Auto-inject platform and platform_id for keytao tools
        if tool_name in ['keytao_create_phrase', 'keytao_submit_batch']:
            if bot and event:
                platform, platform_id = extract_platform_info(bot, event)
                arguments['platform'] = platform
                arguments['platform_id'] = platform_id
                logger.info(f"Auto-injected platform info: {platform}, {platform_id}")
        
        result = await tool_func(**arguments)
        
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Tool {tool_name} execution error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


async def get_openai_response(
    message: str,
    bot: Bot,
    event: Event,
    history: Optional[List[Dict]] = None,
    max_iterations: int = 6
) -> Optional[str]:
    """
    Call Doubao (豆包) API to get response with function calling support
    
    Args:
        message: User message
        bot: Bot instance for context
        event: Event instance for context
        history: Previous conversation history
        max_iterations: Maximum number of function calling iterations (default 6)
    """
    if not ARK_API_KEY:
        return "❌ Doubao API Key 未配置，请联系管理员"
    
    if not AsyncOpenAI:
        return "❌ OpenAI 兼容库未安装，请联系管理员"
    
    try:
        client = AsyncOpenAI(
            api_key=ARK_API_KEY,
            base_url=ARK_BASE_URL,
            timeout=30.0
        )
        
        # Extract platform info
        platform, _ = extract_platform_info(bot, event)
        
        # Build system prompt with platform context
        platform_context = f"\n\n【当前平台信息】\n当前用户使用的平台是: {'Telegram' if platform == 'telegram' else 'QQ' if platform == 'qq' else '未知'}"
        system_prompt_with_context = SYSTEM_PROMPT + platform_context
        
        # Build initial messages with history
        messages = [{"role": "system", "content": system_prompt_with_context}]
        
        # Add conversation history if available with timestamp labels
        if history:
            # Add timestamp labels to help AI understand message recency
            from datetime import datetime
            now = datetime.now()
            
            processed_history = []
            # Calculate which messages are in "recent conversation" (last 6 messages)
            recent_threshold = max(0, len(history) - 6)
            
            for idx, msg in enumerate(history):
                role = msg.get("role")
                content = msg.get("content", "")
                timestamp_str = msg.get("timestamp", "")
                
                # Calculate time difference for logging
                time_diff = None
                if timestamp_str:
                    try:
                        # Parse timestamp (format: 2026-02-19 21:43:33)
                        msg_time = datetime.fromisoformat(timestamp_str.replace(' ', 'T') if ' ' in timestamp_str else timestamp_str)
                        time_diff = (now - msg_time).total_seconds() / 60  # minutes
                    except Exception as e:
                        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
                
                # Use conversation position instead of absolute time for labels
                # This focuses on "recent conversation" rather than "recent time"
                time_label = ""
                if role == "tool":
                    # For tool results, use time-based labels as they're less conversational
                    if time_diff is not None:
                        if time_diff < 3:
                            time_label = "[▶️ 刚刚] "
                        elif time_diff < 10:
                            time_label = f"[⏱️ {int(time_diff)}分钟前] "
                        else:
                            time_label = "[⏰ 较早] "
                # For user/assistant, don't add time labels - they're all part of conversation flow
                # The AI should focus on logical flow, not timestamps
                
                logger.debug(f"Message {idx}: role={role}, time_diff={time_diff:.1f}min, label='{time_label.strip()}'")
                
                # Add label to content
                if time_label:
                    processed_content = time_label + content
                else:
                    processed_content = content
                
                processed_history.append({"role": role, "content": processed_content})
            
            messages.extend(processed_history)
            logger.info(f"Using {len(history)} history messages (last {len(history) - recent_threshold} are recent conversation)")
        
        # Check if user is replying to a message
        reply_context = ""
        
        # Telegram: check for reply_to_message attribute
        reply_to_message = getattr(event, 'reply_to_message', None)
        
        # Note: QQ official API does not provide reply/reference information
        # Even when users quote messages in QQ groups, the bot API doesn't expose it
        
        if reply_to_message:
            # Get bot info
            try:
                bot_info = await bot.get_me()
                bot_id = getattr(bot_info, 'id', None)
            except:
                bot_id = None
            
            # Check who sent the replied message
            reply_from = getattr(reply_to_message, 'from_', None)
            reply_message_text = getattr(reply_to_message, 'text', None)
            
            if reply_from and reply_message_text:
                reply_from_id = getattr(reply_from, 'id', None)
                reply_from_name = getattr(reply_from, 'first_name', '未知用户')
                
                # Check if replying to bot's own message
                is_reply_to_bot = (bot_id and reply_from_id == bot_id)
                
                if is_reply_to_bot:
                    reply_context = f"\n\n【用户正在回复你的消息】\n被引用的消息内容：\n{reply_message_text}\n\n⚠️ 用户的回复是针对这条消息的，请根据这条消息的内容理解用户意图。"
                    logger.info(f"User is replying to bot's message: {reply_message_text[:100]}")
                else:
                    reply_context = f"\n\n【用户正在回复其他人的消息】\n被引用消息的发送者：{reply_from_name}\n被引用的消息内容：\n{reply_message_text}\n\n⚠️ 用户回复的不是你的消息，如果用户说的是操作指令（如'是'、'确认'、'提交'），应该提醒用户：你需要回复bot的消息才能确认操作。"
                    logger.info(f"User is replying to someone else's message (from {reply_from_name})")
        
        # Add current user message with reply context
        user_message_content = message + reply_context
        messages.append({"role": "user", "content": user_message_content})
        
        # Get available tools
        tools = skills_manager.get_tools() if skills_manager.has_tools() else None
        
        # Iterative function calling loop
        for iteration in range(max_iterations):
            # Call AI API
            call_kwargs = {
                "model": ARK_MODEL,
                "messages": messages,
                "max_tokens": ARK_MAX_TOKENS,
                "temperature": ARK_TEMPERATURE,
            }
            
            # Add tools if available
            if tools:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = "auto"
            
            response = await client.chat.completions.create(**call_kwargs)
            
            if not response.choices or len(response.choices) == 0:
                return "呜呜，AI 好像没有回复 qwq 要不再试一次？"
            
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            
            # If no tool calls, return the message
            if finish_reason == "stop" or not choice.message.tool_calls:
                return choice.message.content
            
            # Handle tool calls
            if finish_reason == "tool_calls" and choice.message.tool_calls:
                # Add assistant message with tool calls
                assistant_msg: Dict = {
                    "role": "assistant",
                    "content": choice.message.content
                }
                # Add tool_calls as a separate field
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in choice.message.tool_calls
                ]
                assistant_msg["tool_calls"] = tool_calls_data  # type: ignore
                messages.append(assistant_msg)
                
                # Execute each tool call
                for tool_call in choice.message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Calling tool: {function_name} with args: {function_args}")
                    
                    # 🚨 Smart detection: check if user is confirming a warning but AI didn't pass confirmed=true
                    if function_name == "keytao_create_phrase":
                        confirmed = function_args.get("confirmed", False)
                        
                        # Get user message
                        user_message = event.get_plaintext().strip().lower()
                        
                        # Check for negation keywords FIRST (highest priority)
                        negation_keywords = ["不", "别", "不要", "不用", "取消", "算了", "不行", "不对"]
                        has_negation = any(kw in user_message for kw in negation_keywords)
                        
                        # Only check for confirmation if no negation
                        confirmation_keywords = ["确认", "是", "好", "可以", "同意", "yes", "ok", "确定"]
                        is_confirming = any(kw in user_message for kw in confirmation_keywords)
                        
                        # Auto-fix only if: confirming + not confirmed + NO negation
                        if is_confirming and not confirmed and not has_negation:
                            # Check recent messages for warnings (look back up to 30 messages to ensure we catch previous tool results)
                            had_warning = False
                            check_count = min(30, len(messages))
                            messages_to_check = messages[-check_count:]
                            
                            for idx, msg in enumerate(reversed(messages_to_check)):
                                msg_role = msg.get("role")
                                
                                # Check tool results for requiresConfirmation
                                if msg_role == "tool" and msg.get("content"):
                                    tool_content = msg.get("content", "")
                                    try:
                                        tool_result = json.loads(tool_content)
                                        has_req_confirm = tool_result.get("requiresConfirmation")
                                        has_warnings = tool_result.get("warnings")
                                        if has_req_confirm or has_warnings:
                                            had_warning = True
                                            logger.info(f"🔍 Found warning in tool result (message {idx})")
                                            break
                                    except Exception as e:
                                        pass
                                
                                # Check assistant messages for warning keywords
                                elif msg_role == "assistant" and msg.get("content"):
                                    content = msg.get("content", "")
                                    warning_keywords = ["警告", "确认", "重码", "多编码", "requiresConfirmation"]
                                    found_keywords = [kw for kw in warning_keywords if kw in content]
                                    if found_keywords:
                                        had_warning = True
                                        logger.info(f"🔍 Found warning keywords in assistant message (message {idx}): {found_keywords}")
                                        break
                            
                            if had_warning:
                                logger.error("🚨🚨🚨 CRITICAL: User is confirming a warning but confirmed=false! This will cause infinite loop!")
                                logger.error(f"🚨🚨🚨 User message: '{user_message}'")
                                logger.error(f"🚨🚨🚨 Function args BEFORE auto-fix: {function_args}")
                                
                                # Auto-fix: force confirmed=true to prevent infinite loop
                                function_args["confirmed"] = True
                                logger.warning(f"🔧 AUTO-FIXED: Force set confirmed=true. Function args AFTER: {function_args}")
                                logger.warning("🔧 This is a safety mechanism. AI should learn to pass confirmed=true!")
                    
                    # Call the tool with context
                    function_result = await call_tool_function(function_name, function_args, bot, event)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_result
                    })
                
                # Continue loop to get final response
                continue
            
            # If we reach here, return whatever content we have
            return choice.message.content or "呜呜，AI 好像没有回复 qwq 要不再试一次？"
        
        # Max iterations reached
        return "呜呜，处理太久了 qwq 要不再试一次？"
            
    except Exception as e:
        logger.error(f"Doubao API error: {e}")
        return "呜呜，AI 服务暂时不可用 qwq 等等再来找我吧 ～"


@ai_chat.handle()
async def handle_ai_chat(bot: Bot, event: Event):
    """
    Handle AI chat using DashScope (Qwen) API
    Only triggered when no other handlers match (priority 99)
    """
    # Import platform-specific types
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.telegram.event import GroupMessageEvent as TGGroupMessageEvent
    except ImportError:
        TelegramBot = None
        TGGroupMessageEvent = None
    
    try:
        from nonebot.adapters.qq import Bot as QQBot
        from nonebot.adapters.qq import MessageSegment as QQMessageSegment
        from nonebot.adapters.qq.event import GroupAtMessageCreateEvent, C2CMessageCreateEvent
    except ImportError:
        QQBot = None
        QQMessageSegment = None
        GroupAtMessageCreateEvent = None
        C2CMessageCreateEvent = None
    
    # Get message text
    message_text = event.get_plaintext().strip()
    
    if not message_text:
        await ai_chat.finish("你好呀～ owo 我是喵喵，键道输入法的助手！有什么可以帮你的吗？")
        return
    
    # Get conversation key
    conv_key = get_conversation_key(bot, event)
    
    # Get conversation history
    history = get_history(conv_key)
    
    # Get AI response with context and history (wait for completion before sending)
    response = await get_openai_response(message_text, bot, event, history)
    
    # Handle error response
    if not response:
        await ai_chat.finish("呜呜，处理请求时出错了 qwq 要不再试一次？")
        return
    
    # Save to conversation history
    add_to_history(conv_key, message_text, response)
    
    # Platform-specific reply handling
    # Detect platform by bot class name (more reliable)
    bot_class_name = bot.__class__.__name__
    bot_module_name = bot.__class__.__module__
    
    logger.debug(f"Bot type: {bot_class_name}, Module: {bot_module_name}")
    
    # Telegram: keep URLs (supports links), reply to user message
    if 'telegram' in bot_module_name.lower():
        message_id = getattr(event, 'message_id', None)
        logger.info(f"Telegram message_id: {message_id}")
        if message_id:
            try:
                logger.info(f"Attempting Telegram reply to message_id: {message_id}")
                await bot.send(
                    event=event,
                    message=response,
                    reply_to_message_id=message_id
                )
                logger.info("Telegram reply sent successfully")
                return  # Successfully sent with reply, exit handler
            except Exception as e:
                logger.error(f"Failed to send Telegram reply: {e}", exc_info=True)
                # Fallback to normal send
                await ai_chat.finish(response)
        else:
            logger.warning("Telegram message_id not found, using finish")
            await ai_chat.finish(response)
    
    # QQ: remove URLs (API restriction), try to reply to user message
    elif 'qq' in bot_module_name.lower() or bot_class_name == 'Bot':
        filtered_response = remove_urls(response)
        logger.info(f"QQ platform detected, filtering URLs. Original: {len(response)} chars, Filtered: {len(filtered_response)} chars")
        
        # Try to get QQ message id for reply
        qq_msg_id = getattr(event, 'id', None) or getattr(event, 'message_id', None)
        logger.info(f"QQ message id: {qq_msg_id}")
        
        if qq_msg_id:
            # Method 1: Try using bot.send with msg_id parameter
            try:
                logger.info(f"Attempting QQ reply to message id: {qq_msg_id}")
                await bot.send(
                    event=event,
                    message=filtered_response,
                    msg_id=qq_msg_id
                )
                logger.info("QQ reply sent successfully with msg_id")
                return
            except Exception as e:
                logger.warning(f"Failed to send QQ reply with msg_id: {e}")
            
            # Method 2: Try using event.reply() method
            if hasattr(event, 'reply') and callable(getattr(event, 'reply', None)):
                try:
                    reply_func = getattr(event, 'reply')
                    await reply_func(filtered_response)
                    logger.info("QQ reply sent successfully with event.reply")
                    return
                except Exception as e:
                    logger.warning(f"Failed to use QQ event.reply: {e}")
        
        # Fallback: normal send without reference
        logger.info("QQ falling back to normal send without reply")
        await ai_chat.finish(filtered_response)
    
    # Other platforms: send normally
    else:
        logger.warning(f"Unknown platform, sending without filtering: {bot_class_name}")
        await ai_chat.finish(response)


