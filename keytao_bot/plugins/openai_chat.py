"""
DashScope (Qwen) Chat plugin
使用阿里云通义千问 API 进行智能对话
通过 Skills 系统动态加载工具
"""
import json
from typing import Optional, List, Dict

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

# Get configuration
driver = get_driver()
config = driver.config
DASHSCOPE_API_KEY = getattr(config, "dashscope_api_key", None)
DASHSCOPE_BASE_URL = getattr(config, "dashscope_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL = getattr(config, "dashscope_model", "qwen-plus")
DASHSCOPE_MAX_TOKENS = getattr(config, "dashscope_max_tokens", 1000)
DASHSCOPE_TEMPERATURE = getattr(config, "dashscope_temperature", 0.7)

# Initialize skills manager and load all skills
skills_manager = SkillsManager()
skills_manager.load_all_skills()
logger.info(f"Loaded {len(skills_manager.get_tools())} tools from skills")

# System prompt with compliance requirements
SYSTEM_PROMPT = """你是键道输入法的 AI 助手，负责解答用户关于键道输入法的问题。

【重要原则】

你必须主动调用工具获取准确信息，不要提供猜测性或通用性的回答。

【词语识别和提取】

重要：用户提问时，要从句子中准确提取要查询的词语或编码。

示例：
• "如果 这个词的编码是" → 提取"如果"，调用 keytao_lookup_by_word(word="如果")
• "世界 怎么打" → 提取"世界"，调用 keytao_lookup_by_word(word="世界")
• "中国 用键道怎么输入" → 提取"中国"，调用 keytao_lookup_by_word(word="中国")
• "abc 对应什么" → 提取"abc"，调用 keytao_lookup_by_code(code="abc")
• "nau 是什么词" → 提取"nau"，调用 keytao_lookup_by_code(code="nau")

不要要求用户重新提供词语，直接从他们的问题中提取并查询。

【工具使用指南】

1. 概念性问题（调用 keytao_fetch_docs）：
   • "键道的编码是什么"
   • "键道输入法规则"  
   • "键道怎么学"
   • "键道和五笔的区别"
   • "键道的字根"

2. 按编码查词（调用 keytao_lookup_by_code）：
   • "abc 对应什么词"
   • "nau 是什么"
   • "这个编码 xyz 打出什么"
   → 提取英文字母编码，立即查询

3. 按词查编码（调用 keytao_lookup_by_word）：
   • "你好 怎么打"
   • "世界 的编码"
   • "如何输入 中国"
   • "如果 这个词的编码是"
   → 提取中文词语，立即查询

【回答要求】

• 基于工具返回的实际数据回答，不要编造
• 如果工具查询失败，告知用户查询结果并引导访问官网
• 回答要简洁直接，避免冗长的解释
• 使用纯文本格式（不要用 Markdown 语法）
• 用【】表示标题，用 • 表示列表，用空行分段

【结果展示格式】

查询编码或词条时，必须按以下格式展示：

按词查编码示例（keytao_lookup_by_word）：
查询"如果"的编码：

【查询结果】
词: 如果

编码列表（按使用频率排序）:
1. ri (权重: 100) - 最常用
2. rjgl (权重: 50) - 次常用

如果有多个编码，权重越高表示越常用。

按编码查词示例（keytao_lookup_by_code）：
查询编码"abc"：

【查询结果】
编码: abc

词条列表（按使用频率排序）:
1. 阿爸 (权重: 100) - 最常用
2. 阿伯 (权重: 50) - 次常用

结果展示要点：
• 显示权重值（从工具返回的 weight 字段）
• 按权重从高到低排序（权重高=更常用）
• 标注"最常用"、"次常用"等提示
• 如果只有一个结果，直接显示即可
• 如果没有找到结果，明确告知并引导用户访问官网加词

【资源链接】

• 官网和加词: https://keytao.vercel.app
• 完整文档: https://keytao-docs.vercel.app

【合规要求】

遵守中华人民共和国法律法规，不提供违法违规信息，保护用户隐私。

记住：从用户的问题中提取词语或编码，立即调用工具查询，不要让用户重复提供。"""


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
            # Telegram: in groups, check for mentions
            elif isinstance(event, GroupMessageEvent):
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
                
                logger.debug("Bot not mentioned in group, will not handle")
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


# Create chat handler with custom rule
ai_chat = on_message(rule=should_handle, priority=99, block=True)


async def call_tool_function(tool_name: str, arguments: Dict) -> str:
    """Call a tool function and return result as JSON string"""
    tool_func = skills_manager.get_tool_function(tool_name)
    if not tool_func:
        return json.dumps({"error": f"Tool {tool_name} not found"}, ensure_ascii=False)
    
    try:
        result = await tool_func(**arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Tool {tool_name} execution error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


async def get_openai_response(message: str, max_iterations: int = 3) -> Optional[str]:
    """
    Call DashScope (Qwen) API to get response with function calling support
    
    Args:
        message: User message
        max_iterations: Maximum number of function calling iterations
    """
    if not DASHSCOPE_API_KEY:
        return "❌ DashScope API Key 未配置，请联系管理员"
    
    if not AsyncOpenAI:
        return "❌ OpenAI 兼容库未安装，请联系管理员"
    
    try:
        client = AsyncOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=30.0
        )
        
        # Build initial messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
        
        # Get available tools
        tools = skills_manager.get_tools() if skills_manager.has_tools() else None
        
        # Iterative function calling loop
        for iteration in range(max_iterations):
            # Call AI API
            call_kwargs = {
                "model": DASHSCOPE_MODEL,
                "messages": messages,
                "max_tokens": DASHSCOPE_MAX_TOKENS,
                "temperature": DASHSCOPE_TEMPERATURE,
            }
            
            # Add tools if available
            if tools:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = "auto"
            
            response = await client.chat.completions.create(**call_kwargs)
            
            if not response.choices or len(response.choices) == 0:
                return "❌ AI 未返回有效响应"
            
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
                    
                    # Call the tool
                    function_result = await call_tool_function(function_name, function_args)
                    
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
            return choice.message.content or "❌ AI 未返回有效响应"
        
        # Max iterations reached
        return "❌ AI 处理超时，请重试"
            
    except Exception as e:
        logger.error(f"DashScope API error: {e}")
        return f"❌ AI 服务暂时不可用，请稍后重试"


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
        await ai_chat.finish("你好！我是 AI 助手，有什么我可以帮你的吗？")
        return
    
    # Get AI response (wait for completion before sending)
    response = await get_openai_response(message_text)
    
    # Handle error response
    if not response:
        await ai_chat.finish("❌ 抱歉，处理请求时出错了")
        return
    
    # Platform-specific reply handling
    try:
        # Telegram group: use reply_to_message_id
        if TelegramBot and TGGroupMessageEvent and isinstance(bot, TelegramBot) and isinstance(event, TGGroupMessageEvent):
            message_id = event.message_id
            await bot.send(
                event=event,
                message=response,
                reply_to_message_id=message_id
            )
            raise FinishedException
        
        # Other platforms: send normally (QQ doesn't support message reference in official API)
        else:
            await ai_chat.finish(response)
            
    except FinishedException:
        raise
    except Exception as e:
        logger.error(f"Error sending reply: {e}")
        # Fallback to normal send
        await ai_chat.finish(response)


