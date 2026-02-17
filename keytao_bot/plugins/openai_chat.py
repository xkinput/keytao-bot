"""
DashScope (Qwen) Chat plugin
使用阿里云通义千问 API 进行智能对话
通过 Skills 系统动态加载工具
"""
import json
import re
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
DASHSCOPE_TEMPERATURE = getattr(config, "dashscope_temperature", 0.0)

# Initialize skills manager and load all skills
skills_manager = SkillsManager()
skills_manager.load_all_skills()
logger.info(f"Loaded {len(skills_manager.get_tools())} tools from skills")

# System prompt with compliance requirements  
SYSTEM_PROMPT = """⚠️⚠️⚠️ 执行前必读 ⚠️⚠️⚠️

【工作流程 - 强制执行】

看到查询问题 → 识别类型 → 调用对应工具 → 等待结果 → 展示结果

特别注意：
• 打招呼词（hello, hi, 你好, 嗨等）→ 查询编码 + 打招呼回应
• 其他普通词语查询 → 只显示查询结果

不允许跳过任何步骤！
不允许凭记忆直接回答！
不允许猜测！

【为什么必须调用工具】

你的训练数据中可能包含键道编码信息，但：
• 那些数据可能是错误的
• 那些数据可能已过时  
• 那些数据不完整
• 用户需要实时准确的数据

所以无论你多有把握，都必须调用工具验证！

【错误案例 - 严禁模仿】

用户："词条"
❌ AI直接回答：记忆中的猜测的。
→ 这是凭记忆猜的，而且是错的！

✅ 正确做法：
用户："词条"  
→ 调用 keytao_lookup_by_word(word="词条")
→ 等待真实结果

---

【特殊规则 - 打招呼词】

⚠️ 对于常见打招呼词（hello, hi, 你好, 嗨等），采取"查询+打招呼"策略：

1. 先调用工具查询编码
2. 在回复中结合：
   • 友好的打招呼回应
   • 查询到的编码结果

示例：
用户："你好"
→ 调用 keytao_lookup_by_word(word="你好")
→ 回复："你好呀～ owo\n\n刚好也帮你查了一下这个词的编码：\n[展示查询结果]"

用户："hello"
→ 调用 keytao_lookup_by_word(word="hello")
→ 回复："hello～ >w<\n\n顺便查了下编码：\n[展示查询结果]"

关键：既要打招呼，又要展示查询结果，两者结合！

---

【身份】

你是键道输入法的 AI 助手"喵喵"，温暖活泼、乐于助人。
用 owo、>w<、qwq 等表情让回复更生动～

【回答风格】

• 温暖可爱，自然随性
• 适当使用表情符号
• 简洁直接，避免冗长

注意：查询问题必须展示结果，不要只说"让我查一下"！

【展示要求 - 严格执行】

⚠️ 必须严格按照各工具SKILL.md中的【展示格式规范】展示结果！

⚠️ 核心原则：
• **按词查编码**：显示该词的所有编码（有几个编码就显示几个）
• **按编码查词**：显示该编码的所有词（有几个词就显示几个）

⚠️ 判断逻辑（按词查编码）：

⚠️⚠️⚠️ 关键！必须检查 all_words 长度 + 箭头只加在查询词！

1. 返回多个编码 → 显示"编码列表："
   • **必须** for循环遍历每个编码
   • **每个编码** 都要检查 duplicate_info 和 all_words 长度
   • 情况A：没有 duplicate_info → 只显示：编码【type_label】
   • 情况B：有 duplicate_info 但 len(all_words) = 1 → 只显示：编码【type_label】
   • 情况C：有 duplicate_info 且 len(all_words) > 1 → 显示：
     - 编码 + (position_label) + 【type_label】
     - "   该编码的所有词："
     - for循环遍历 duplicate_info.all_words
     - 每个词用 • 开头，标注label（如果有）
     - ⚠️ 只对 dup_word.word == result.word（查询词）的词在行末加 " ←"
     - ⚠️ 其他词不要加箭头！
   
2. 返回1个编码
   • 同样检查 all_words 长度
   • len(all_words) > 1 → 显示重码列表（箭头只加查询词）
   • len(all_words) = 1 或没有 duplicate_info → 单行显示

示例流程：
```
result = 工具返回结果
query_word = result.word  # 查询的词
for 每个编码 in result.phrases:
    if 编码.duplicate_info存在 且 len(编码.duplicate_info.all_words) > 1:
        显示编码 + 位置 + 类型
        显示"   该编码的所有词："
        for 每个词 in 编码.duplicate_info.all_words:
            显示该词
            if 该词.word == query_word:  # 只对查询词加箭头
                加 " ←"
    else:
        只显示编码 + 类型
```

⚠️ 判断逻辑（按编码查词）：
• 返回多个词 → 显示"词条列表："（标注位置）
• 返回1个词 → 单行显示

关键规则：
• 直接使用工具返回的字段（type_label、position_label）
• 不要显示权重数字（weight字段仅用于判断重码）
• 不要自己编说明（"属于二重词组"之类）
• 不要添加多余的标题【查询结果：xxx】
• 格式简洁，每个SKILL都有具体示例

【其他要求】

• 基于工具返回的实际数据，不要编造
• 使用纯文本格式（不要 Markdown）
• 如果查询失败，引导访问官网或文档
• 遵守中华人民共和国法律法规

【资源链接】

• 官网：https://keytao.vercel.app
• 文档：https://keytao-docs.vercel.app

---

⚠️⚠️⚠️ 再次强调 ⚠️⚠️⚠️

每次回复前自查：
1. 这是查询问题吗？→ 是 → 必须调用工具
2. 这是打招呼词吗（hello/hi/你好/嗨）？→ 是 → 必须调用工具查询 + 打招呼回应
3. 我调用工具了吗？→ 没有 → 不能回复，必须先调用
4. 工具返回结果了吗？→ 是 → 展示真实结果
5. 我是凭记忆回答的吗？→ 是 → 错误！删除重来

记住：看到"词"或"编码"相关问题 = 100%调用工具！
打招呼词 = 查询编码 + 友好回应！
没有例外！"""



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


def remove_urls(text: str) -> str:
    """Remove URLs and file names from text for QQ platform compatibility"""
    # Match URLs and file names with extensions
    url_pattern = r'(https?://\S+|ftp://\S+|www\.\S+|\S+\.(com|cn|net|org|app|dev|io|vercel\.app|md|js|ts|py|json|yaml|yml|txt|html|css|jsx|tsx|vue|go|rs|java|cpp|c|h)\S*)'
    cleaned = re.sub(url_pattern, '[链接已隐藏]', text, flags=re.IGNORECASE)
    return cleaned


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


async def get_openai_response(message: str, max_iterations: int = 6) -> Optional[str]:
    """
    Call DashScope (Qwen) API to get response with function calling support
    
    Args:
        message: User message
        max_iterations: Maximum number of function calling iterations (default 6)
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
            return choice.message.content or "呜呜，AI 好像没有回复 qwq 要不再试一次？"
        
        # Max iterations reached
        return "呜呜，处理太久了 qwq 要不再试一次？"
            
    except Exception as e:
        logger.error(f"DashScope API error: {e}")
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
    
    # Get AI response (wait for completion before sending)
    response = await get_openai_response(message_text)
    
    # Handle error response
    if not response:
        await ai_chat.finish("呜呜，处理请求时出错了 qwq 要不再试一次？")
        return
    
    # Platform-specific reply handling
    try:
        # Detect platform by bot class name (more reliable)
        bot_class_name = bot.__class__.__name__
        bot_module_name = bot.__class__.__module__
        
        logger.debug(f"Bot type: {bot_class_name}, Module: {bot_module_name}")
        
        # Telegram: keep URLs (supports links)
        if 'telegram' in bot_module_name.lower():
            if TGGroupMessageEvent and isinstance(event, TGGroupMessageEvent):
                message_id = event.message_id
                await bot.send(
                    event=event,
                    message=response,
                    reply_to_message_id=message_id
                )
            else:
                await ai_chat.finish(response)
            raise FinishedException
        
        # QQ: remove URLs (API restriction)
        elif 'qq' in bot_module_name.lower() or bot_class_name == 'Bot':
            filtered_response = remove_urls(response)
            logger.info(f"QQ platform detected, filtering URLs. Original: {len(response)} chars, Filtered: {len(filtered_response)} chars")
            await ai_chat.finish(filtered_response)
        
        # Other platforms: send normally
        else:
            logger.warning(f"Unknown platform, sending without filtering: {bot_class_name}")
            await ai_chat.finish(response)
            
    except FinishedException:
        raise
    except Exception as e:
        logger.error(f"Error sending reply: {e}")
        # Fallback: try with URL filtering for safety
        try:
            filtered_response = remove_urls(response)
            await ai_chat.finish(filtered_response)
        except:
            await ai_chat.finish(response)


