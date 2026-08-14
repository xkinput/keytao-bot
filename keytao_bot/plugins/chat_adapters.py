"""Extract platform context and adapt image and reply inputs for chat."""

import asyncio
import re
from dataclasses import dataclass
from itertools import islice
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import nonebot
from nonebot import get_driver
from nonebot.adapters import Bot, Event
from nonebot.log import logger

from ..harness.conversation import ConversationAddress
from ..utils.image_input import (
    ImageAttachment,
    VisionConfigurationError,
    VisionProxyResult,
    VisionRuntimeConfig,
    VisionServiceError,
    extract_image_attachments,
    request_vision_description,
)
from ..utils.llm_policy import log_chat_usage
from ..utils.memory_store import ChatMemoryContext
from .chat_render import _split_telegram_text


try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    logger.warning("openai package not installed, OpenAI chat plugin will not work")


async def _send_telegram_plain_chunks(
    bot: Bot,
    event: Event,
    text: str,
    *,
    reply_to_message_id: Optional[int] = None,
) -> None:
    chunks = _split_telegram_text(text)
    for index, chunk in enumerate(chunks):
        kwargs: Dict[str, Any] = {"event": event, "message": chunk}
        if index == 0 and reply_to_message_id:
            kwargs["reply_to_message_id"] = reply_to_message_id
        await bot.send(**kwargs)


@dataclass(frozen=True)
class ReplyReferenceInfo:
    is_reply: bool = False
    is_to_bot: bool = False
    sender_id: str = ""
    sender_name: str = ""
    text: str = ""
    mentioned_user_ids: Tuple[str, ...] = ()
    images: Tuple[ImageAttachment, ...] = ()


try:
    driver = get_driver()
except ValueError:
    nonebot.init()
    driver = get_driver()
config = driver.config


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


OPENAI_API_KEY = (
    getattr(config, "openai_api_key", None)
    or getattr(config, "gemini_api_key", None)
    or getattr(config, "ark_api_key", None)
)


OPENAI_BASE_URL = (
    getattr(config, "openai_base_url", None)
    or getattr(config, "gemini_base_url", None)
    or getattr(config, "ark_base_url", None)
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
)


OPENAI_MODEL = (
    getattr(config, "openai_model", None)
    or getattr(config, "gemini_model", None)
    or getattr(config, "ark_model", None)
    or "gemini-2.0-flash"
)


OPENAI_MAX_TOKENS: int = _as_int((
    getattr(config, "openai_max_tokens", None)
    or getattr(config, "gemini_max_tokens", None)
    or getattr(config, "ark_max_tokens", None)
    or 1000
), 1000)


openai_timeout_value = getattr(config, "openai_timeout", None)


if openai_timeout_value is None:
    openai_timeout_value = getattr(config, "gemini_timeout", None)


if openai_timeout_value is None:
    openai_timeout_value = getattr(config, "ark_timeout", None)


if openai_timeout_value is None:
    openai_timeout_value = 180.0


OPENAI_TIMEOUT: float = _as_float(openai_timeout_value, 180.0)


openai_temperature_value = getattr(config, "openai_temperature", None)


if openai_temperature_value is None:
    openai_temperature_value = getattr(config, "gemini_temperature", None)


if openai_temperature_value is None:
    openai_temperature_value = getattr(config, "ark_temperature", None)


if openai_temperature_value is None:
    openai_temperature_value = 0.7


OPENAI_TEMPERATURE: float = _as_float(openai_temperature_value, 0.7)


VISION_CONFIG = VisionRuntimeConfig(
    enabled=_as_bool(getattr(config, "vision_enabled", None), False),
    api_key=str(getattr(config, "vision_api_key", None) or "").strip(),
    base_url=str(getattr(config, "vision_base_url", None) or "").strip(),
    model=str(getattr(config, "vision_model", None) or "").strip(),
    timeout=max(
        5.0,
        min(_as_float(getattr(config, "vision_timeout", None), 60.0), 180.0),
    ),
    max_tokens=max(
        128,
        min(_as_int(getattr(config, "vision_max_tokens", None), 1200), 8000),
    ),
    max_images=max(
        1,
        min(_as_int(getattr(config, "vision_max_images", None), 3), 8),
    ),
    max_image_bytes=max(
        256 * 1024,
        min(
            _as_int(getattr(config, "vision_max_image_bytes", None), 5 * 1024 * 1024),
            20 * 1024 * 1024,
        ),
    ),
    max_total_image_bytes=max(
        512 * 1024,
        min(
            _as_int(
                getattr(config, "vision_max_total_image_bytes", None),
                12 * 1024 * 1024,
            ),
            40 * 1024 * 1024,
        ),
    ),
    max_image_pixels=max(
        65_536,
        min(
            _as_int(getattr(config, "vision_max_image_pixels", None), 2_621_440),
            16_777_216,
        ),
    ),
    max_total_image_pixels=max(
        65_536,
        min(
            _as_int(
                getattr(config, "vision_max_total_image_pixels", None),
                7_864_320,
            ),
            50_331_648,
        ),
    ),
    qq_napcat_source_root=str(
        getattr(config, "vision_qq_napcat_source_root", None)
        or "/app/.config/QQ"
    ).strip(),
    qq_napcat_mapped_root=str(
        getattr(config, "vision_qq_napcat_mapped_root", None)
        or "/app/napcat/qq"
    ).strip(),
)


VISION_MAX_CONCURRENT_REQUESTS = max(
    1,
    min(
        _as_int(getattr(config, "vision_max_concurrent_requests", None), 2),
        8,
    ),
)


_vision_request_semaphore = asyncio.Semaphore(VISION_MAX_CONCURRENT_REQUESTS)


def extract_platform_info(bot: Bot, event: Event) -> Tuple[str, str]:
    """Extract platform type and user ID from event."""
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.onebot.v11 import Bot as QQBot
    except ImportError:
        TelegramBot = None
        QQBot = None

    if TelegramBot and isinstance(bot, TelegramBot):
        from_ = getattr(event, 'from_', None)
        user_id = str(getattr(from_, 'id', '')) if from_ else ''
        return ("telegram", user_id)
    elif QQBot and isinstance(bot, QQBot):
        user_id = str(getattr(event, 'user_id', ''))
        return ("qq", user_id)
    else:
        logger.warning(f"Unknown platform: {bot.__class__.__name__}")
        return ("unknown", "")


def _display_name_from_telegram_user(user: object) -> str:
    first_name = str(getattr(user, 'first_name', '') or '').strip()
    last_name = str(getattr(user, 'last_name', '') or '').strip()
    username = str(getattr(user, 'username', '') or '').strip()
    full_name = " ".join(part for part in (first_name, last_name) if part)
    return full_name or username


def _display_name_from_qq_sender(sender: object, fallback: str) -> str:
    for field in ("card", "nickname"):
        value = None
        if isinstance(sender, dict):
            value = sender.get(field)
        else:
            value = getattr(sender, field, None)
        text = str(value or "").strip()
        if text:
            return text
    for dump_method in ("model_dump", "dict"):
        dump = getattr(sender, dump_method, None)
        if not callable(dump):
            continue
        try:
            data = dump()
        except Exception:
            continue
        if isinstance(data, dict):
            for field in ("card", "nickname"):
                text = str(data.get(field) or "").strip()
                if text:
                    return text
    if isinstance(sender, dict):
        text = str(sender.get('user_id') or "").strip()
        if text and text != str(fallback):
            return text
    return str(fallback or "").strip()


def _telegram_conversation_space_id(chat: object, event: object) -> str:
    """Keep Telegram forum topics isolated inside the same supergroup."""
    chat_id = str(getattr(chat, "id", "") or "").strip()
    thread_id = str(
        getattr(event, "message_thread_id", "")
        or getattr(getattr(event, "message", None), "message_thread_id", "")
        or ""
    ).strip()
    if chat_id and thread_id:
        return f"{chat_id}:thread:{thread_id}"
    return chat_id


async def extract_memory_context(
    bot: Bot,
    event: Event,
    reply_info: Optional[ReplyReferenceInfo] = None,
) -> ChatMemoryContext:
    """Extract actor, space and reply-target metadata for scoped memory."""
    platform, user_id = extract_platform_info(bot, event)
    space_type = "private"
    space_id = user_id
    speaker_name = ""
    target_user_id = ""
    target_name = "喵喵"

    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.telegram.event import GroupMessageEvent as TelegramGroupMessageEvent
    except ImportError:
        TelegramBot = None
        TelegramGroupMessageEvent = None
    try:
        from nonebot.adapters.onebot.v11 import Bot as QQBot
        from nonebot.adapters.onebot.v11.event import GroupMessageEvent as QQGroupMessageEvent
    except ImportError:
        QQBot = None
        QQGroupMessageEvent = None

    if TelegramBot and isinstance(bot, TelegramBot):
        from_ = getattr(event, 'from_', None)
        speaker_name = _display_name_from_telegram_user(from_) if from_ else ""
        chat = getattr(event, 'chat', None)
        if TelegramGroupMessageEvent and isinstance(event, TelegramGroupMessageEvent):
            space_type = "group"
            space_id = _telegram_conversation_space_id(chat, event)
        elif chat is not None:
            space_id = str(getattr(chat, 'id', '') or user_id)

        if reply_info is not None:
            target_user_id = reply_info.sender_id
            target_name = reply_info.sender_name or target_name
        else:
            reply_to_message = getattr(event, 'reply_to_message', None)
            reply_from = getattr(reply_to_message, 'from_', None) if reply_to_message else None
            if reply_from:
                target_user_id = str(getattr(reply_from, 'id', '') or "")
                target_name = _display_name_from_telegram_user(reply_from) or target_user_id

    elif QQBot and isinstance(bot, QQBot):
        sender = getattr(event, 'sender', None)
        speaker_name = _display_name_from_qq_sender(sender, user_id)
        if QQGroupMessageEvent and isinstance(event, QQGroupMessageEvent):
            space_type = "group"
            space_id = str(getattr(event, 'group_id', '') or "")

        reply_message_id = extract_onebot_reply_id(event)
        if reply_info is not None:
            target_user_id = reply_info.sender_id
            target_name = reply_info.sender_name or target_name
        elif reply_message_id:
            try:
                reply_payload = await bot.get_msg(message_id=int(reply_message_id))
                reply_sender = reply_payload.get('sender', {}) if isinstance(reply_payload, dict) else {}
                target_user_id = str(reply_sender.get('user_id') or reply_payload.get('user_id', ''))
                target_name = _display_name_from_qq_sender(reply_sender, target_user_id)
            except Exception as error:
                logger.debug(f"Failed to extract OneBot reply target {reply_message_id}: {error}")

    return ChatMemoryContext(
        platform=platform,
        user_id=user_id,
        space_type=space_type,
        space_id=space_id or user_id,
        speaker_name=speaker_name or user_id,
        target_user_id=target_user_id,
        target_name=target_name,
    )


def _space_key_from_memory_context(memory_context: ChatMemoryContext) -> Tuple[str, str]:
    return (memory_context.platform, memory_context.space_scope_id)


def extract_onebot_reply_id(event: Event) -> Optional[str]:
    """Extract replied message id from OneBot v11 message segments."""
    try:
        message_to_check = getattr(event, 'original_message', None) or getattr(event, 'message', None)
        if not message_to_check:
            return None
        for segment in islice(iter(message_to_check), 64):
            segment_type = getattr(segment, 'type', None)
            segment_data = getattr(segment, 'data', {})
            if segment_type == 'reply':
                reply_id = segment_data.get('id') or segment_data.get('message_id')
                if reply_id is not None:
                    return str(reply_id)
    except Exception as error:
        logger.debug(f"Failed to extract OneBot reply id: {error}")
    return None


def extract_onebot_mentioned_user_ids(message: object) -> Tuple[str, ...]:
    """Extract explicit @ user ids from a OneBot message payload."""
    mentioned_user_ids: List[str] = []

    if isinstance(message, str):
        for match in re.finditer(r"\[CQ:at,qq=([^,\]]+)", message):
            qq = match.group(1).strip()
            if qq and qq.lower() != "all":
                mentioned_user_ids.append(qq)
        return tuple(mentioned_user_ids)

    try:
        for segment in message:  # type: ignore
            if isinstance(segment, dict):
                seg_type = segment.get('type')
                seg_data = segment.get('data', {})
            else:
                seg_type = getattr(segment, 'type', None)
                seg_data = getattr(segment, 'data', {})
            if seg_type != 'at':
                continue
            qq = str(
                seg_data.get('qq')
                or seg_data.get('user_id')
                or seg_data.get('id')
                or ""
            ).strip()
            if qq and qq.lower() != "all":
                mentioned_user_ids.append(qq)
    except Exception:
        pass

    return tuple(mentioned_user_ids)


def extract_onebot_plaintext(message: object) -> str:
    """Extract plain text from OneBot message payload."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message.strip()

    extract_fn = getattr(message, 'extract_plain_text', None)
    if callable(extract_fn):
        try:
            return str(extract_fn()).strip()
        except Exception:
            pass

    parts: List[str] = []
    try:
        for segment in message:  # type: ignore
            if isinstance(segment, dict):
                seg_type = segment.get('type')
                seg_data = segment.get('data', {})
            else:
                seg_type = getattr(segment, 'type', None)
                seg_data = getattr(segment, 'data', {})
            if seg_type == 'text':
                text = seg_data.get('text', '')
                if text:
                    parts.append(str(text))
            elif seg_type == 'at':
                qq = str(seg_data.get('qq') or seg_data.get('user_id') or "").strip()
                if qq and qq.lower() != "all":
                    parts.append(f"@{qq} ")
    except Exception:
        pass
    return ''.join(parts).strip()


def _build_qq_reply_message(
    qq_message_segment: object,
    reply_message_id: object,
    target_user_id: str,
    text: str,
    mention_target: bool,
) -> object:
    """Build a QQ reply, optionally mentioning the target user first."""
    message = qq_message_segment.reply(reply_message_id)
    if mention_target and target_user_id:
        message = message + qq_message_segment.at(target_user_id) + " "
    return message + text


async def extract_reply_reference_info(bot: Bot, event: Event) -> ReplyReferenceInfo:
    """Extract replied-message metadata for Telegram and OneBot v11."""
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
            return ReplyReferenceInfo()
        reply_message = (
            getattr(reply_to_message, 'original_message', None)
            or getattr(reply_to_message, 'message', None)
        )
        reply_images = extract_image_attachments(
            reply_message,
            "telegram",
            source="reply",
        )
        try:
            bot_info = await bot.get_me()
            bot_id = str(getattr(bot_info, 'id', '') or '')
        except Exception:
            bot_id = ""

        reply_from = getattr(reply_to_message, 'from_', None)
        reply_text = (
            extract_onebot_plaintext(reply_message)
            or getattr(reply_to_message, 'text', None)
            or getattr(reply_to_message, 'caption', None)
            or ""
        )
        if not reply_from:
            return ReplyReferenceInfo(
                is_reply=True,
                text=str(reply_text or "").strip(),
                images=reply_images,
            )
        reply_from_id = str(getattr(reply_from, 'id', '') or '')
        reply_from_name = _display_name_from_telegram_user(reply_from) or reply_from_id or "未知用户"
        return ReplyReferenceInfo(
            is_reply=True,
            is_to_bot=bool(bot_id and reply_from_id == bot_id),
            sender_id=reply_from_id,
            sender_name=reply_from_name,
            text=str(reply_text or "").strip(),
            images=reply_images,
        )

    if QQBot and isinstance(bot, QQBot):
        reply_message_id = extract_onebot_reply_id(event)
        if not reply_message_id:
            return ReplyReferenceInfo()
        logger.info(f"Detected OneBot reply segment, reply message_id: {reply_message_id}")
        try:
            reply_payload = await bot.get_msg(message_id=int(reply_message_id))
        except Exception as error:
            logger.warning(f"Failed to fetch replied OneBot message {reply_message_id}: {error}")
            return ReplyReferenceInfo(is_reply=True)

        sender = reply_payload.get('sender', {}) if isinstance(reply_payload, dict) else {}
        reply_from_id = str(sender.get('user_id') or reply_payload.get('user_id', ''))
        reply_from_name = _display_name_from_qq_sender(sender, reply_from_id or '未知用户')
        reply_message = reply_payload.get('message') if isinstance(reply_payload, dict) else None
        reply_text = extract_onebot_plaintext(reply_message)
        mentioned_user_ids = extract_onebot_mentioned_user_ids(reply_message)
        reply_images = extract_image_attachments(
            reply_message,
            "qq",
            source="reply",
        )
        if not reply_text and isinstance(reply_payload, dict):
            reply_text = str(reply_payload.get('raw_message', '')).strip()
        if not mentioned_user_ids and isinstance(reply_payload, dict):
            mentioned_user_ids = extract_onebot_mentioned_user_ids(
                str(reply_payload.get('raw_message', '') or "")
            )

        bot_id = str(getattr(bot, 'self_id', ''))
        return ReplyReferenceInfo(
            is_reply=True,
            is_to_bot=bool(bot_id and reply_from_id == bot_id),
            sender_id=reply_from_id,
            sender_name=reply_from_name,
            text=reply_text,
            mentioned_user_ids=mentioned_user_ids,
            images=reply_images,
        )

    return ReplyReferenceInfo()


async def build_reply_context(
    bot: Bot,
    event: Event,
    reply_info: Optional[ReplyReferenceInfo] = None,
) -> str:
    """Build reply context for Telegram and OneBot v11."""
    info = reply_info or await extract_reply_reference_info(bot, event)
    if not info.is_reply or not info.text:
        return ""

    if info.is_to_bot:
        return (
            f"\n\n【用户正在回复你的消息】\n被引用的消息内容：\n{info.text}\n\n"
            "⚠️ 用户的回复是针对这条消息的，请根据这条消息的内容理解用户意图。"
        )

    return (
        f"\n\n【用户正在回复其他人的消息】\n被引用消息的发送者：{info.sender_name or '未知用户'}\n"
        f"被引用的消息内容：\n{info.text}\n\n"
        "⚠️ 用户回复的不是你的消息，如果用户说的是操作指令（如'是'、'确认'、'提交'），"
        "应该提醒用户：你需要回复bot的消息才能确认操作。"
    )


def extract_event_image_attachments(
    event: Event,
    platform: str,
) -> Tuple[ImageAttachment, ...]:
    """Extract current-message images before adapter to-me preprocessing can remove segments."""

    message = (
        getattr(event, "original_message", None)
        or getattr(event, "message", None)
    )
    return extract_image_attachments(message, platform, source="current")


def event_may_reference_images(event: Event, platform: str) -> bool:
    """Detect reply-image candidates without making platform API calls."""

    if platform == "qq":
        return extract_onebot_reply_id(event) is not None
    if platform != "telegram":
        return False
    reply_to_message = getattr(event, "reply_to_message", None)
    if not reply_to_message:
        return False
    reply_message = (
        getattr(reply_to_message, "original_message", None)
        or getattr(reply_to_message, "message", None)
    )
    return bool(extract_image_attachments(reply_message, "telegram", source="reply"))


async def _describe_images_for_deepseek_in_slot(
    bot: Bot,
    attachments: Tuple[ImageAttachment, ...],
    user_prompt: str,
) -> VisionProxyResult:
    """Run the vision request while the caller owns its timeout/concurrency slot."""

    if not AsyncOpenAI:
        raise VisionConfigurationError("OpenAI-compatible SDK is unavailable")
    VISION_CONFIG.validate()
    async with AsyncOpenAI(
        api_key=VISION_CONFIG.api_key,
        base_url=VISION_CONFIG.base_url,
        timeout=VISION_CONFIG.timeout,
        max_retries=0,
    ) as client:
        result = await request_vision_description(
            client,
            bot,
            attachments,
            user_prompt,
            VISION_CONFIG,
        )
    log_chat_usage(
        logger,
        result.response,
        operation="vision_proxy",
        model=VISION_CONFIG.model,
    )
    logger.info(
        "Vision proxy completed: "
        f"model={VISION_CONFIG.model} images={result.image_count} "
        f"warnings={len(result.warnings)} description_len={len(result.description)}"
    )
    return result


async def _describe_images_for_deepseek(
    bot: Bot,
    attachments: Tuple[ImageAttachment, ...],
    user_prompt: str,
) -> VisionProxyResult:
    """Run the full independent vision path under one local deadline and slot."""

    try:
        async with asyncio.timeout(VISION_CONFIG.timeout):
            async with _vision_request_semaphore:
                return await _describe_images_for_deepseek_in_slot(
                    bot,
                    attachments,
                    user_prompt,
                )
    except TimeoutError as error:
        raise VisionServiceError("vision processing timed out") from error


def _vision_unavailable_reply() -> str:
    return (
        "我收到图片了，但当前主模型 DeepSeek V4 Flash 是纯文本模型，"
        "独立图片理解服务还没有启用，所以这次不能可靠地看图。"
        "请管理员配置视觉代理后再试，避免我假装看到了图片。"
    )


def _vision_input_failed_reply() -> str:
    return (
        "图片已收到，但下载、大小或格式校验没有通过，因此没有发送给视觉服务。"
        "请改用 JPG、PNG 或 WEBP，并尽量控制在 5 MB 以内再试。"
    )


def _vision_service_failed_reply() -> str:
    return "图片理解服务这次没有返回完整结果，请稍后重试；我不会凭空猜图片内容。"
