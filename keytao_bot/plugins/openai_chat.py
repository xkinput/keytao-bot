"""
DashScope (Qwen) Chat plugin
使用阿里云通义千问 API 进行智能对话
通过 Skills 系统动态加载工具
"""
import json
import asyncio
from typing import Optional, List, Dict

from nonebot import on_message, get_driver
from nonebot.adapters import Bot, Event
from nonebot.rule import to_me
from nonebot.log import logger

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

# Message auto-recall configuration (in seconds, 0 = disabled)
AUTO_RECALL_DELAY = getattr(config, "auto_recall_delay", 0)

# Initialize skills manager and load all skills
skills_manager = SkillsManager()
skills_manager.load_all_skills()
logger.info(f"Loaded {len(skills_manager.get_tools())} tools from skills")

# System prompt with compliance requirements
SYSTEM_PROMPT = """你是一个友善、专业的 AI 助手。你的回答必须遵守以下规范：

1. 法律合规：
   - 严格遵守中华人民共和国相关法律法规
   - 不提供任何违法违规的信息或建议
   - 不参与任何危害国家安全、社会公共利益的讨论
   - 尊重知识产权，不提供盗版、侵权内容

2. 内容规范：
   - 不得包含色情、暴力、恐怖等不良内容
   - 不得传播虚假信息、谣言
   - 不得发表歧视性、侮辱性言论
   - 不得教唆、煽动违法犯罪行为

3. 伦理价值观：
   - 践行社会主义核心价值观
   - 尊重人的尊严与基本人权
   - 倡导和平、友善、包容的人道主义精神
   - 保护未成年人身心健康

4. 回答原则：
   - 提供客观、准确、有帮助的信息
   - 对于敏感话题保持中立和理性
   - 不确定的信息要明确告知
   - 拒绝回答可能造成危害的问题

5. 隐私保护：
   - 不收集、存储用户个人隐私信息
   - 不泄露对话中的敏感信息
   - 尊重用户隐私权

如果你有可用的工具（functions），在需要时主动调用它们来提供更准确的信息。
请在遵守以上规范的前提下，友好、专业地回答用户问题。"""

# Create chat handler with lower priority (after commands)
ai_chat = on_message(rule=to_me(), priority=99, block=True)


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
    # Get message text
    message_text = event.get_plaintext().strip()
    
    if not message_text:
        await ai_chat.finish("你好！我是 AI 助手，有什么我可以帮你的吗？")
        return
    
    # Show typing indicator
    typing_msg = await ai_chat.send("🤔 思考中...")
    
    # Get AI response
    response = await get_openai_response(message_text)
    
    # Check adapter type
    from nonebot.adapters.telegram import Bot as TelegramBot, MessageEvent as TelegramMessageEvent
    from nonebot.adapters.qq import Bot as QQBot
    
    if isinstance(bot, TelegramBot):
        # Telegram: edit typing message to final response (works for both private and group chats)
        if typing_msg and hasattr(typing_msg, 'message_id') and hasattr(typing_msg, 'chat'):
            try:
                logger.debug(f"Editing Telegram message {typing_msg.message_id} in chat {typing_msg.chat.id}")
                await bot.edit_message_text(
                    chat_id=typing_msg.chat.id,
                    message_id=typing_msg.message_id,
                    text=response
                )
                logger.info(f"Successfully edited message in {'group' if isinstance(event, TelegramMessageEvent) and event.chat.type in ['group', 'supergroup'] else 'private'} chat")
            except Exception as e:
                logger.error(f"Failed to edit Telegram message: {e}")
                # Fallback: send as new message
                await bot.send(event, response)
        else:
            logger.warning(f"Cannot edit message: missing attributes (has message_id: {hasattr(typing_msg, 'message_id')}, has chat: {hasattr(typing_msg, 'chat')})")
            await bot.send(event, response)
    
    elif isinstance(bot, QQBot):
        # QQ: send new message and optionally recall
        reply_msg = await bot.send(event, response)
        
        # Auto-recall both typing and reply messages if enabled
        if AUTO_RECALL_DELAY > 0:
            if typing_msg and hasattr(typing_msg, 'id'):
                asyncio.create_task(_schedule_recall_qq(bot, event, typing_msg.id, AUTO_RECALL_DELAY))
            if reply_msg and hasattr(reply_msg, 'id'):
                asyncio.create_task(_schedule_recall_qq(bot, event, reply_msg.id, AUTO_RECALL_DELAY))
    
    else:
        # Other adapters: just send
        await bot.send(event, response)


async def _schedule_recall_qq(bot, event: Event, message_id: str, delay: int):
    """Schedule QQ message recall after delay"""
    try:
        await asyncio.sleep(delay)
        
        # Import event types
        from nonebot.adapters.qq import C2CMessageCreateEvent, GroupAtMessageCreateEvent
        
        # QQ adapter: different recall methods for C2C and Group
        if isinstance(event, C2CMessageCreateEvent):
            await bot.delete_c2c_message(openid=event.author.id, message_id=message_id)
            logger.debug(f"Recalled C2C message {message_id}")
        elif isinstance(event, GroupAtMessageCreateEvent):
            await bot.delete_group_message(group_openid=event.group_openid, message_id=message_id)
            logger.debug(f"Recalled Group message {message_id}")
        else:
            logger.warning(f"Auto-recall not supported for event type: {type(event)}")
    except Exception as e:
        logger.error(f"Failed to recall message {message_id}: {e}")
