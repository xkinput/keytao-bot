"""
Account binding plugin
Handles /bind command for platform account binding
"""
from nonebot import on_command, get_driver
from nonebot.adapters import Bot, Event
from nonebot.exception import FinishedException
from nonebot.log import logger
from nonebot.rule import Rule
import httpx

# Get configuration from NoneBot
driver = get_driver()
config = driver.config
KEYTAO_API_BASE = getattr(config, "keytao_api_base", "https://keytao.vercel.app")
BOT_API_TOKEN = getattr(config, "bot_api_token", None)

# Debug log
logger.info(f"[account_bind] KEYTAO_API_BASE: {KEYTAO_API_BASE}")
logger.info(f"[account_bind] BOT_API_TOKEN loaded: {bool(BOT_API_TOKEN)}")

GROUP_TRIGGER_KEYWORDS = ("键道", "喵喵")



# Custom rule for handling /bind only in appropriate contexts
async def should_handle_bind(bot: Bot, event: Event) -> bool:
    """
    Rule to handle /bind command:
    - Private messages: always
    - Group messages: only when bot is mentioned or replied to
    """
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.telegram.event import PrivateMessageEvent, GroupMessageEvent
        from nonebot.adapters.onebot.v11 import Bot as QQBot
        from nonebot.adapters.onebot.v11.event import PrivateMessageEvent as QQPrivateMessageEvent, GroupMessageEvent as QQGroupMessageEvent
        
        if isinstance(bot, TelegramBot):
            # Telegram: always in private
            if isinstance(event, PrivateMessageEvent):
                return True
            # Telegram: in group, check for mention or reply
            elif isinstance(event, GroupMessageEvent):
                # Check if message is a reply to bot
                reply_to_message = getattr(event, 'reply_to_message', None)
                if reply_to_message:
                    bot_info = await bot.get_me()
                    # Check if the replied message is from the bot
                    reply_from = getattr(reply_to_message, 'from_', None)
                    if reply_from and reply_from.id == bot_info.id:
                        logger.info("[account_bind] Message is a reply to bot, will handle")
                        return True
                
                # Check for @mention
                bot_info = await bot.get_me()
                bot_username = bot_info.username
                message_to_check = getattr(event, 'original_message', event.message)
                message_text = event.get_plaintext().strip()
                
                for segment in message_to_check:
                    if segment.type == 'mention':
                        mention_text = segment.data.get('text', '')
                        if mention_text == f"@{bot_username}":
                            logger.info(f"[account_bind] Bot mentioned in group, will handle")
                            return True

                if any(keyword in message_text for keyword in GROUP_TRIGGER_KEYWORDS):
                    logger.info("[account_bind] Group message contains trigger keyword, will handle")
                    return True
                
                logger.debug("[account_bind] Bot not mentioned/replied in group, will not handle")
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
                    logger.info("[account_bind] QQ group message contains trigger keyword, will handle")
                    return True

                return False

            # QQ: fallback to default to_me() behavior for other event types
            from nonebot.rule import to_me
            return await to_me()(bot, event, {})
        
        else:
            # Other platforms: use to_me()
            from nonebot.rule import to_me
            return await to_me()(bot, event, {})
            
    except Exception as e:
        logger.error(f"[account_bind] Error in should_handle_bind rule: {e}")
        # Fallback: allow in all cases to avoid breaking the command
        return True


# Bind command with custom rule
bind_cmd = on_command("bind", rule=Rule(should_handle_bind), priority=5, block=True)


@bind_cmd.handle()
async def handle_bind(bot: Bot, event: Event):
    """
    Handle bind command
    Usage: /bind AB12CD
    """
    # Log execution info
    logger.info(f"[account_bind] Triggered by bot: {bot.__class__.__name__}, user: {event.get_user_id()}")
    
    # Get platform info
    try:
        from nonebot.adapters.onebot.v11 import Bot as QQBot
        from nonebot.adapters.telegram import Bot as TelegramBot
        
        if isinstance(bot, QQBot):
            platform = "qq"
            platform_id = event.get_user_id()
        elif isinstance(bot, TelegramBot):
            platform = "telegram"
            platform_id = event.get_user_id()
        else:
            await bind_cmd.finish("不支持的平台")
            return
    except Exception as e:
        logger.error(f"Get platform info error: {e}")
        await bind_cmd.finish("获取平台信息失败")
        return

    # Get bind key from message
    message_text = event.get_plaintext().strip()
    parts = message_text.split()
    
    if len(parts) < 2:
        help_text = (
            "📝 如何绑定机器人账号：\n\n"
            "1. 登录键道网站：https://keytao.vercel.app\n"
            "2. 点击网站右上角的用户名，进入【我的资料】页面\n"
            "   （ 或直接访问：https://keytao.vercel.app/profile ）\n"
            "3. 在【机器人账号绑定】区域点击【生成绑定码】\n"
            "4. 复制生成的绑定码\n"
            "5. 在这里发送：/bind [你的绑定码]\n\n"
            "示例：/bind AB12CD\n\n"
            "💡 提示：如果在群聊中，需要 @我 或回复我的消息"
        )
        await bind_cmd.finish(help_text)
        return
    
    key = parts[1].upper()

    # Check if BOT_API_TOKEN is configured
    if not BOT_API_TOKEN:
        logger.error("BOT_API_TOKEN not configured")
        await bind_cmd.finish("❌ 机器人配置错误，请联系管理员")
        return

    # Call verify API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{KEYTAO_API_BASE}/api/auth/link/verify",
                headers={
                    "X-Bot-Token": BOT_API_TOKEN,
                    "Content-Type": "application/json"
                },
                json={
                    "key": key,
                    "platform": platform,
                    "platformId": platform_id
                },
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    user_name = data.get("userName", "")
                    nickname = data.get("userNickname") or user_name
                    await bind_cmd.finish(
                        f"✅ 绑定成功！\n\n"
                        f"账号：{nickname}\n"
                        f"现在你可以使用机器人创建词条了～ >w<"
                    )
                else:
                    await bind_cmd.finish(f"❌ {data.get('message', '绑定失败')}")
            else:
                try:
                    error_data = response.json()
                    message = error_data.get("message", "绑定失败")
                except:
                    message = "绑定失败"
                await bind_cmd.finish(f"❌ {message}")

    except httpx.TimeoutException:
        await bind_cmd.finish("❌ 请求超时，请稍后重试")
    except FinishedException:
        raise  # Let NoneBot handle this, don't catch it
    except Exception as e:
        logger.error(f"Bind error: {e}")
        await bind_cmd.finish("❌ 绑定失败，请稍后重试")
