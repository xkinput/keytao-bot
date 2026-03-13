"""
OpenAI-compatible chat plugin
使用 OpenAI 兼容 API 进行智能对话
通过 Skills 系统动态加载工具
"""
import json
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
OPENAI_API_KEY = getattr(config, "openai_api_key", None) or getattr(config, "gemini_api_key", None) or getattr(config, "ark_api_key", None)
OPENAI_BASE_URL = (
    getattr(config, "openai_base_url", None)
    or getattr(config, "gemini_base_url", None)
    or getattr(config, "ark_base_url", None)
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
)
OPENAI_MODEL = getattr(config, "openai_model", None) or getattr(config, "gemini_model", None) or getattr(config, "ark_model", None) or "gemini-2.0-flash"
OPENAI_MAX_TOKENS = getattr(config, "openai_max_tokens", None) or getattr(config, "gemini_max_tokens", None) or getattr(config, "ark_max_tokens", None) or 1000
OPENAI_TEMPERATURE = getattr(config, "openai_temperature", None)
if OPENAI_TEMPERATURE is None:
    OPENAI_TEMPERATURE = getattr(config, "gemini_temperature", None)
if OPENAI_TEMPERATURE is None:
    OPENAI_TEMPERATURE = getattr(config, "ark_temperature", None)
if OPENAI_TEMPERATURE is None:
    OPENAI_TEMPERATURE = 0.7

GROUP_TRIGGER_KEYWORDS = ("键道", "喵喵")

# Initialize skills manager and load all skills
skills_manager = SkillsManager()
skills_manager.load_all_skills()
logger.info(f"Loaded {len(skills_manager.get_tools())} tools from skills")

# Initialize history store (SQLite)
history_store = get_history_store()
MAX_HISTORY_MESSAGES = 30  # Keep last 30 messages (15 rounds) for batch operations

# Pending confirmation state: stores tool args waiting for user to confirm a warning
# Key: (platform, user_id), Value: {"function": str, "args": dict}
pending_confirmations: Dict[Tuple[str, str], Dict] = {}

# System prompt with compliance requirements  
# 精简的核心原则（约60行）
SYSTEM_PROMPT_CORE = """你是键道输入法的AI助手"喵喵"。

你的定位不是冷冰冰的流程机器人，而是一个会聊天、会理解语境、也真的能帮用户处理键道相关事情的助手。
你要像一个聪明、自然、反应快的人类助手一样说话：该聊天时聊天，该办事时办事，该调用工具时就果断调用工具。

━━━━━━━━━━━━━━━━━━━━━
⚠️ 核心原则（必须遵守)
━━━━━━━━━━━━━━━━━━━━━

1. 立即执行原则
   • ⚠️ 只处理标有 [当前请求] 的消息！
   • 标有 [历史] 的消息是已经处理完的历史记录，绝对不要重复处理！
   • 用户说操作词（添加/删除/修改）→ 立即调用工具
   • 用户说"确认/是" + 最近有警告 → 立即调用confirmed=true
   • ⚠️ "确认/好/是/可以"不是提交指令！提交草稿必须用户明确说"提交"/"提审"/"发起审核"
   • 不要多余查询！不要反复询问！

2. 确认流程（防止无限循环）
   第1次调用 → 返回警告 → 告知用户 → 询问确认
   用户确认 → 立即再次调用 + confirmed=true（相同参数）
   ⚠️ 如果不传confirmed=true会无限循环！

   Delete 操作不会阻塞，写入草稿后成功响应中包含 notes（被删除的词条信息），必须告知用户。
   
3. 多步操作原则
   用户要求连续多个步骤（删A→删B→加C→加D）时：
   • 按顺序连续调用工具，不要在步骤之间停下来等用户确认
   • Delete 步骤永远不停，直接继续
   • 只有 Create/Change 返回 requiresConfirmation 时才暂停询问
   • 每次工具返回后，用 draft_snapshot 展示当前草稿状态
   • 全部完成后一次性汇报结果 + 最终草稿状态

3. 否定词识别（避免误操作）
   否定词：不、别、不要、不用、取消、算了
   • 有否定词 → 取消当前操作
   • "不 + 操作" → 停止
   • "不 + 操作，新需求" → 取消当前 + 执行新需求

4. 必须调用工具
   • 查询词/编码 → 调用查询工具
   • 询问规则/文档 → 调用文档工具
   • 创建/删除/修改 → 调用创建工具
   • 不允许凭记忆回答！

5. 聊天与查词的边界
        • 你不仅是工具助手，也要会正常聊天、接话、解释、陪聊、开玩笑
        • 用户在闲聊、提问感受、求安慰、玩梗、讲故事、问建议时 → 直接自然回复，不要强行调用工具
        • 用户如果明显在问“这个词怎么打”“这个编码是什么词”“规则是什么”“帮我增删改词条” → 调用对应工具
        • 用户如果只是丢来一个词、短语、单字、编码，也不要机械套规则，要结合上下文判断他是在查词还是在聊天
        • 短消息不等于查词，也不等于闲聊。你要像真人一样理解语气、语境、上下文延续关系
        • 只有在你判断这是键道查询、文档查询、词库操作时，才调用工具；否则就正常聊天

6. 理解示例（按这个思路判断）
     • 用户说：你好呀
         → 这是打招呼，直接聊天回复，不调用工具
     • 用户说：我今天有点烦
         → 这是聊天/倾诉，直接安慰或接话，不调用工具
     • 用户说：讲个笑话
         → 这是娱乐聊天，直接回复，不调用工具
     • 用户说：找寻
         → 大概率是在问这个词怎么打，调用按词查编码工具
     • 用户说：fzxw
         → 大概率是在问这个编码是什么词，调用按编码查词工具
     • 用户说：不是怎么打
         → 明确在查编码，调用按词查编码工具
     • 用户说：这个规则我没看懂
         → 明确在问文档/规则，调用文档工具
     • 用户说：帮我加个词
         → 明确是词库操作，调用创建工具
     • 用户说：不是，我是在开玩笑
         → 这是对上文的澄清，按聊天理解，不要误触发工具

7. 历史消息处理
   • 只关注最近3-5轮对话
   • 用户说"确认" → 检查最近是否有待确认操作
   • 用户提新需求 → 开始新对话

8. 链接输出规则
   • ⚠️ 所有平台（包括QQ）都必须输出完整URL，禁止将链接替换为[链接]或任何占位符

9. 工具调用格式规则
    • 工具调用只能通过 API 提供的原生 tool_calls 机制发起
    • 绝对不要在回复文本中手写工具名、JSON 参数、XML/HTML 标签
    • 绝对不要输出类似 <function=xxx {...}></function> 的内容
    • 如果需要调用工具，直接发起结构化 tool call，不要解释调用过程

━━━━━━━━━━━━━━━━━━━━━
回复风格
━━━━━━━━━━━━━━━━━━━━━

• 温暖自然，简洁直接
• 可以适度活泼，但不要每句都堆表情
• 聊天时要像真人，不要满嘴流程术语，不要机械重复规则
• 不要动不动就教育用户，除非用户真的在问规则或操作
• 查询必须展示结果
• 使用纯文本格式（不要Markdown）
• 不同信息分段，空行隔开
"""



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
        from nonebot.adapters.onebot.v11 import Bot as QQBot
        from nonebot.adapters.onebot.v11.event import PrivateMessageEvent as QQPrivateMessageEvent, GroupMessageEvent as QQGroupMessageEvent
        
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

                if any(keyword in message_text for keyword in GROUP_TRIGGER_KEYWORDS):
                    logger.info("Group message contains trigger keyword, will handle")
                    return True
                
                logger.debug("Bot not mentioned/replied in group, will not handle")
                return False
            return False
        
        elif isinstance(bot, QQBot):
            if isinstance(event, QQPrivateMessageEvent):
                return True
            if isinstance(event, QQGroupMessageEvent):
                from nonebot.rule import to_me

                if await to_me()(bot, event, {}):
                    return True

                message_text = event.get_plaintext().strip()
                if any(keyword in message_text for keyword in GROUP_TRIGGER_KEYWORDS):
                    logger.info("QQ group message contains trigger keyword, will handle")
                    return True

                return False

            # QQ: fallback to default to_me() behavior for other event types
            from nonebot.rule import to_me
            return await to_me()(bot, event, {})
        
        else:
            # Other platforms: use to_me() by default
            from nonebot.rule import to_me
            return await to_me()(bot, event, {})
            
    except Exception as e:
        logger.error(f"Error in should_handle rule: {e}")
        return False

# Clear history command
from nonebot import on_command
from nonebot.rule import Rule
clear_cmd = on_command("clear", aliases={"清空对话", "清空历史"}, rule=Rule(should_handle), priority=5, block=True)

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


def extract_onebot_reply_id(event: Event) -> Optional[str]:
    """
    Extract replied message id from OneBot v11 message segments
    从 OneBot v11 消息段中提取被回复的消息 ID
    """
    try:
        message_to_check = getattr(event, 'original_message', None) or getattr(event, 'message', None)
        if not message_to_check:
            return None

        for segment in message_to_check:
            segment_type = getattr(segment, 'type', None)
            segment_data = getattr(segment, 'data', {})

            if segment_type == 'reply':
                reply_id = segment_data.get('id') or segment_data.get('message_id')
                if reply_id is not None:
                    return str(reply_id)
    except Exception as error:
        logger.debug(f"Failed to extract OneBot reply id: {error}")

    return None


def extract_onebot_plaintext(message: object) -> str:
    """
    Extract plain text from OneBot message payload
    从 OneBot 消息内容中提取纯文本
    """
    if message is None:
        return ""

    if isinstance(message, str):
        return message.strip()

    extract_plain_text = getattr(message, 'extract_plain_text', None)
    if callable(extract_plain_text):
        try:
            plain_text = extract_plain_text()
            return str(plain_text).strip()
        except Exception:
            pass

    parts: List[str] = []

    try:
        for segment in message:  # type: ignore
            if isinstance(segment, dict):
                segment_type = segment.get('type')
                segment_data = segment.get('data', {})
            else:
                segment_type = getattr(segment, 'type', None)
                segment_data = getattr(segment, 'data', {})

            if segment_type == 'text':
                text = segment_data.get('text', '')
                if text:
                    parts.append(str(text))
    except Exception:
        pass

    return ''.join(parts).strip()


async def build_reply_context(bot: Bot, event: Event) -> str:
    """
    Build reply context for Telegram and OneBot v11
    为 Telegram 和 OneBot v11 构建回复上下文
    """
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
    except ImportError:
        TelegramBot = None

    try:
        from nonebot.adapters.onebot.v11 import Bot as QQBot
    except ImportError:
        QQBot = None

    if TelegramBot and isinstance(bot, TelegramBot):
        reply_to_message = getattr(event, 'reply_to_message', None)
        if not reply_to_message:
            return ""

        try:
            bot_info = await bot.get_me()
            bot_id = getattr(bot_info, 'id', None)
        except Exception:
            bot_id = None

        reply_from = getattr(reply_to_message, 'from_', None)
        reply_message_text = getattr(reply_to_message, 'text', None)

        if reply_from and reply_message_text:
            reply_from_id = getattr(reply_from, 'id', None)
            reply_from_name = getattr(reply_from, 'first_name', '未知用户')
            is_reply_to_bot = bool(bot_id and reply_from_id == bot_id)

            if is_reply_to_bot:
                logger.info(f"User is replying to bot's message: {reply_message_text[:100]}")
                return f"\n\n【用户正在回复你的消息】\n被引用的消息内容：\n{reply_message_text}\n\n⚠️ 用户的回复是针对这条消息的，请根据这条消息的内容理解用户意图。"

            logger.info(f"User is replying to someone else's message (from {reply_from_name})")
            return f"\n\n【用户正在回复其他人的消息】\n被引用消息的发送者：{reply_from_name}\n被引用的消息内容：\n{reply_message_text}\n\n⚠️ 用户回复的不是你的消息，如果用户说的是操作指令（如'是'、'确认'、'提交'），应该提醒用户：你需要回复bot的消息才能确认操作。"

        return ""

    if QQBot and isinstance(bot, QQBot):
        reply_message_id = extract_onebot_reply_id(event)
        if not reply_message_id:
            return ""

        logger.info(f"Detected OneBot reply segment, reply message_id: {reply_message_id}")

        try:
            reply_payload = await bot.get_msg(message_id=int(reply_message_id))
        except Exception as error:
            logger.warning(f"Failed to fetch replied OneBot message {reply_message_id}: {error}")
            return ""

        sender = reply_payload.get('sender', {}) if isinstance(reply_payload, dict) else {}
        reply_from_id = str(sender.get('user_id') or reply_payload.get('user_id', ''))
        reply_from_name = sender.get('card') or sender.get('nickname') or reply_from_id or '未知用户'
        reply_message_text = extract_onebot_plaintext(
            reply_payload.get('message') if isinstance(reply_payload, dict) else None
        )

        if not reply_message_text and isinstance(reply_payload, dict):
            reply_message_text = str(reply_payload.get('raw_message', '')).strip()

        if not reply_message_text:
            logger.info(f"Replied OneBot message {reply_message_id} has no plain text content")
            return ""

        bot_id = str(getattr(bot, 'self_id', ''))
        is_reply_to_bot = bool(bot_id and reply_from_id == bot_id)

        if is_reply_to_bot:
            logger.info(f"User is replying to bot's QQ message: {reply_message_text[:100]}")
            return f"\n\n【用户正在回复你的消息】\n被引用的消息内容：\n{reply_message_text}\n\n⚠️ 用户的回复是针对这条消息的，请根据这条消息的内容理解用户意图。"

        logger.info(f"User is replying to someone else's QQ message (from {reply_from_name})")
        return f"\n\n【用户正在回复其他人的消息】\n被引用消息的发送者：{reply_from_name}\n被引用的消息内容：\n{reply_message_text}\n\n⚠️ 用户回复的不是你的消息，如果用户说的是操作指令（如'是'、'确认'、'提交'），应该提醒用户：你需要回复bot的消息才能确认操作。"

    return ""


def extract_platform_info(bot: Bot, event: Event) -> tuple[str, str]:
    """
    Extract platform type and user ID from event
    提取平台类型和用户ID
    
    Returns:
        (platform, platform_id): tuple of platform name and user ID
    """
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.onebot.v11 import Bot as QQBot
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
        # QQ platform (OneBot v11 via NapCat)
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
        if tool_name in ['keytao_create_phrase', 'keytao_submit_batch', 'keytao_list_draft_items', 'keytao_remove_draft_item', 'keytao_batch_add_to_draft', 'keytao_batch_remove_draft_items', 'keytao_recall_batch']:
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
    max_iterations: int = 20
) -> Optional[str]:
    """
    Call OpenAI-compatible API to get response with function calling support
    
    Args:
        message: User message
        bot: Bot instance for context
        event: Event instance for context
        history: Previous conversation history
        max_iterations: Maximum number of function calling iterations (default 20)
    """
    if not OPENAI_API_KEY:
        return "❌ OpenAI 兼容 API Key 未配置，请联系管理员"
    
    if not AsyncOpenAI:
        return "❌ OpenAI 兼容库未安装，请联系管理员"
    
    try:
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=30.0
        )
        
        # Extract platform info
        platform, _ = extract_platform_info(bot, event)
        
        # Build system prompt dynamically: Core + Platform + Skills
        platform_context = f"\n\n【当前平台信息】\n当前用户使用的平台是: {'Telegram' if platform == 'telegram' else 'QQ' if platform == 'qq' else '未知'}"
        skill_instructions = skills_manager.get_skill_instructions()
        
        system_prompt_full = SYSTEM_PROMPT_CORE + platform_context + skill_instructions
        
        logger.info(f"📋 System prompt length: {len(system_prompt_full)} chars")
        
        # Build initial messages with history
        messages = [{"role": "system", "content": system_prompt_full}]
        
        # Add conversation history, marking past user messages as [历史] so AI won't re-execute them
        if history:
            processed_history = []
            for msg in history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    processed_history.append({"role": role, "content": f"[历史] {content}"})
                else:
                    processed_history.append({"role": role, "content": content})
            
            messages.extend(processed_history)
            logger.info(f"Using {len(history)} history messages (marked as [历史])")
        
        # Detect: user is confirming a pending warning
        # If current message is a short confirmation AND recent history has a warning, inject explicit instruction
        confirmation_words = {"确认", "是", "好", "可以", "同意", "yes", "ok", "确定", "嗯", "行"}
        user_msg_lower = message.strip().lower()
        is_short_confirm = user_msg_lower in confirmation_words or (len(message.strip()) <= 4 and any(w in user_msg_lower for w in confirmation_words))
        
        pending_confirm_hint = ""
        if is_short_confirm and messages:
            # Check recent messages for a pending warning
            for msg in reversed(messages):
                if msg.get("role") == "tool":
                    try:
                        tool_result = json.loads(msg.get("content", "{}"))
                        if tool_result.get("requiresConfirmation"):
                            pending_confirm_hint = "\n\n[系统检测：用户在确认上一条警告！请立即用相同参数再次调用keytao_create_phrase，但添加confirmed=true，不要询问用户！]"
                            logger.info("🎯 Detected confirmation of pending warning, injecting hint")
                            break
                    except Exception:
                        pass
                elif msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if any(kw in content for kw in ["警告", "重码", "多编码", "是否确认", "requiresConfirmation"]):
                        pending_confirm_hint = "\n\n[系统检测：用户在确认上一条警告！请立即用相同参数再次调用keytao_create_phrase，但添加confirmed=true，不要询问用户！]"
                        logger.info("🎯 Detected confirmation of pending warning (from assistant msg), injecting hint")
                        break
                    break  # Only check the most recent assistant message
        
        # Check if user is replying to a message
        reply_context = await build_reply_context(bot, event)
        
        # Add current user message with reply context and optional confirmation hint
        # Mark current message clearly so AI only processes this one
        user_message_content = f"[当前请求] {message}" + reply_context + pending_confirm_hint
        messages.append({"role": "user", "content": user_message_content})
        
        # Get available tools
        tools = skills_manager.get_tools() if skills_manager.has_tools() else None
        
        # Iterative function calling loop
        for iteration in range(max_iterations):
            # Call AI API
            call_kwargs = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "max_tokens": OPENAI_MAX_TOKENS,
                "temperature": OPENAI_TEMPERATURE,
            }
            
            # Add tools if available
            if tools:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = "auto"

            logger.info(f"Calling model: {OPENAI_MODEL} (iteration {iteration + 1}/{max_iterations})")
            
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
                    if function_name == "keytao_create_phrase" and bot and event:
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
                    
                    # If tool requires confirmation, save pending state so next user message can bypass AI
                    if function_name in ("keytao_create_phrase", "keytao_submit_batch") and bot and event:
                        try:
                            result_data = json.loads(function_result)
                            if result_data.get("requiresConfirmation"):
                                platform, user_id = extract_platform_info(bot, event)
                                conv_key = (platform, user_id)
                                # Save args without confirmed flag for retry
                                saved_args = {k: v for k, v in function_args.items() if k != "confirmed"}
                                pending_confirmations[conv_key] = {"function": function_name, "args": saved_args}
                                logger.info(f"💾 Saved pending confirmation for {conv_key}: {saved_args}")
                        except Exception:
                            pass
                    
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
        logger.error(f"API error: {e}")
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
        from nonebot.adapters.onebot.v11 import Bot as QQBot
        from nonebot.adapters.onebot.v11 import MessageSegment as QQMessageSegment
    except ImportError:
        QQBot = None
        QQMessageSegment = None
    
    # Get message text
    message_text = event.get_plaintext().strip()
    
    if not message_text:
        await ai_chat.finish("你好呀～ owo 我是喵喵，键道输入法的助手！有什么可以帮你的吗？")
        return
    
    # Get conversation key
    conv_key = get_conversation_key(bot, event)
    
    # Check if user is confirming a pending warning — bypass AI entirely
    confirmation_words = {"确认", "是", "好", "可以", "同意", "yes", "ok", "确定", "嗯", "行", "/confirm"}
    negation_words = {"不", "别", "不要", "不用", "取消", "算了", "不行", "不对"}
    msg_lower = message_text.strip().lower()
    is_confirming = msg_lower in confirmation_words or (len(message_text) <= 4 and any(w in msg_lower for w in confirmation_words))
    has_negation = any(w in msg_lower for w in negation_words)
    
    if is_confirming and not has_negation and conv_key in pending_confirmations:
        pending = pending_confirmations.pop(conv_key)
        func_name = pending["function"]
        func_args = {**pending["args"], "confirmed": True}
        logger.info(f"✅ Direct confirmation bypass: calling {func_name} with confirmed=True, args={func_args}")
        
        result_json = await call_tool_function(func_name, func_args, bot, event)
        try:
            result = json.loads(result_json)
            if result.get("success"):
                response = f"✅ 已确认添加到草稿批次！\n\n• 词：{func_args.get('word', '')}\n• 编码：{func_args.get('code', '')}\n\n是否立即提交审核？回复'提交'或'是'即可～\n也可以继续添加/修改/删除词条哦 owo"
            else:
                response = f"操作失败：{result.get('message', '未知错误')} qwq"
        except Exception:
            response = "操作完成 owo"
    elif has_negation and conv_key in pending_confirmations:
        pending_confirmations.pop(conv_key)
        logger.info(f"🚫 User negated pending confirmation for {conv_key}, clearing state")
        response = None  # let AI handle the negation normally
    else:
        response = None
    
    if response is None:
        # Get AI response with context and history (wait for completion before sending)
        history = get_history(conv_key)
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
    
    # QQ: reply to user message
    elif 'onebot' in bot_module_name.lower() or bot_class_name == 'Bot':
        # Try to get QQ message id for reply
        qq_msg_id = getattr(event, 'message_id', None)
        logger.info(f"QQ message_id: {qq_msg_id}")
        
        if qq_msg_id and QQMessageSegment:
            try:
                logger.info(f"Attempting OneBot v11 reply to message_id: {qq_msg_id}")
                await bot.send(
                    event=event,
                    message=QQMessageSegment.reply(qq_msg_id) + response,
                )
                logger.info("OneBot v11 reply sent successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to send OneBot v11 reply: {e}")
        
        # Fallback: normal send without reference
        logger.info("QQ falling back to normal send without reply")
        await ai_chat.finish(response)
    
    # Other platforms: send normally
    else:
        logger.warning(f"Unknown platform, sending without filtering: {bot_class_name}")
        await ai_chat.finish(response)


