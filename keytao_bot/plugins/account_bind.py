"""
Account binding plugin
Handles /bind command for platform account binding
"""
from nonebot import on_command, get_driver
from nonebot.adapters import Bot, Event
from nonebot.exception import FinishedException
from nonebot.log import logger
import httpx

# Get configuration from NoneBot
driver = get_driver()
config = driver.config
KEYTAO_API_BASE = getattr(config, "keytao_api_base", "https://keytao.vercel.app")
BOT_API_TOKEN = getattr(config, "bot_api_token", None)

# Debug log
logger.info(f"[account_bind] KEYTAO_API_BASE: {KEYTAO_API_BASE}")
logger.info(f"[account_bind] BOT_API_TOKEN loaded: {bool(BOT_API_TOKEN)}")

# Bind command
bind_cmd = on_command("bind", priority=5, block=True)


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
        from nonebot.adapters.qq import Bot as QQBot
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
        await bind_cmd.finish(
            "📝 用法：/bind 绑定码\n\n"
            "请先在键道官网生成绑定码：\n"
            "1. 登录 keytao.vercel.app\n"
            "2. 进入【我的资料】页面\n"
            "3. 在【机器人账号绑定】部分点击【生成绑定码】\n"
            "4. 复制绑定码，在这里发送 /bind [绑定码]\n\n"
            "示例：/bind AB12CD"
        )
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
