"""
OpenAI-compatible chat plugin
Uses a Python-side state machine for reliable confirmation handling.
AI handles: chat, queries, tool calling, formatting.
Python handles: confirmation routing, direct execution of simple confirms.
"""
import json
import re
from typing import Any, Optional, List, Dict, Tuple

from nonebot import on_message, on_command, get_driver
from nonebot.adapters import Bot, Event
from nonebot.rule import Rule, to_me
from nonebot.log import logger

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    logger.warning("openai package not installed, OpenAI chat plugin will not work")

from ..skills import SkillsManager
from ..harness.orchestrator import AgentOrchestrator, AgentRequestContext, AgentRuntimeConfig
from ..harness.state import (
    MemoryConversationStateStore,
    PendingAddWord,
    PendingState,
    PendingToolConfirm,
)
from ..harness.tools import ToolContext, ToolExecutor
from ..utils.history_store import get_history_store


# ---------------------------------------------------------------------------
# Message formatting helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown syntax for plain-text platforms (QQ)."""
    text = re.sub(r'```[\w]*\n?(.*?)```', lambda m: m.group(1).strip(), text, flags=re.DOTALL)
    text = re.sub(r'`([^`\n]+)`', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'__(.*?)__', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_((?!\s).*?(?<!\s))_', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


_MV2_RE = re.compile(r'([\\_%*\[\]()~`>#+\-=|{}.!])')


def _escape_mv2_segment(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 in a plain-text segment."""
    return _MV2_RE.sub(r'\\\1', text)


def _to_markdownv2(text: str) -> str:
    """Convert common markdown to Telegram MarkdownV2."""
    result: list[str] = []
    last = 0
    for m in re.finditer(r'```[\w]*\n?.*?```|`[^`\n]+`', text, re.DOTALL):
        result.append(_escape_mv2_segment(text[last:m.start()]))
        result.append(m.group())
        last = m.end()
    result.append(_escape_mv2_segment(text[last:]))
    return ''.join(result)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BIND_HELP_TEXT = (
    "你还没有绑定键道账号哦～\n\n"
    "📝 绑定步骤：\n\n"
    "1. 登录键道网站：https://keytao.vercel.app\n"
    "2. 点击右上角用户名，进入【我的资料】\n"
    "   （或直接访问：https://keytao.vercel.app/profile ）\n"
    "3. 在【机器人账号绑定】区域点击【生成绑定码】\n"
    "4. 复制绑定码\n"
    "5. 在这里发送：/bind [你的绑定码]\n\n"
    "示例：/bind AB12CD\n\n"
    "💡 群聊中需要 @我 或回复我的消息才能触发绑定"
)

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

GROUP_TRIGGER_KEYWORD_START = "键道"
GROUP_TRIGGER_KEYWORD_ANY = "喵喵"
_LEADING_COMMAND_PREFIX_RE = re.compile(
    r"^(?:@\S+|键道|喵喵)[\s:：，,]*",
    re.IGNORECASE,
)
_CLEAR_COMMAND_RE = re.compile(r"^/?(?:clear|清空对话|清空历史)$", re.IGNORECASE)
_PURE_CHINESE_WORDS_RE = re.compile(r'^[\u4e00-\u9fff]+(?:[\s、，,；;]+[\u4e00-\u9fff]+)*$')
_WORD_QUERY_STOPWORDS = ("什么", "怎么", "为何", "为啥", "意思", "含义", "吗", "呢", "呀", "啊", "吧", "嘛")


def _strip_command_message_prefixes(message_text: str) -> str:
    text = message_text.strip()
    while text:
        stripped = _LEADING_COMMAND_PREFIX_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    return text


def _is_clear_command_text(message_text: str) -> bool:
    command_text = _strip_command_message_prefixes(message_text)
    return any(_CLEAR_COMMAND_RE.fullmatch(token) for token in command_text.split())


def _extract_pure_chinese_words(message_text: str) -> List[str]:
    """Extract standalone Chinese words from a simple word-only message."""
    text = message_text.strip()
    if not text or not _PURE_CHINESE_WORDS_RE.fullmatch(text):
        return []
    if any(stopword in text for stopword in _WORD_QUERY_STOPWORDS):
        return []
    return [token for token in re.split(r'[\s、，,；;]+', text) if token]


# ---------------------------------------------------------------------------
# Skills & History
# ---------------------------------------------------------------------------

skills_manager = SkillsManager()
skills_manager.load_all_skills()
logger.info(f"Loaded {len(skills_manager.get_tools())} tools from skills")

history_store = get_history_store()
MAX_HISTORY_MESSAGES = 16


# ---------------------------------------------------------------------------
# Conversation State Machine
# ---------------------------------------------------------------------------

# Per-conversation state: (platform, user_id) -> state
conversation_state_store = MemoryConversationStateStore()
conversation_states: Dict[Tuple[str, str], PendingState] = conversation_state_store.states

CONFIRM_WORDS = frozenset({
    "确认", "是", "好", "好的", "可以", "同意", "yes", "ok", "确定", "嗯", "行", "y",
})
_CONFIRM_MESSAGE_RE = re.compile(
    r"^(?:"
    r"确认|是|好|好的|可以|同意|yes|ok|确定|嗯|行|y"
    r")(?:[\s,，。.!！~吧啦喔哦呀哈呗]*)$",
    re.IGNORECASE,
)
CANCEL_WORDS = frozenset({
    "别", "不要", "不要了", "不用", "不用了", "取消", "算了", "不行", "先不", "先不了", "no", "n",
})
_CANCEL_MESSAGE_RE = re.compile(
    r"^(?:"
    r"取消|算了|别|不行|"
    r"不要(?:了|加了)?|"
    r"不用(?:了)?|"
    r"先不(?:了)?|"
    r"no|n"
    r")(?:[\s,，。.!！~吧啦喔哦呀哈呗]*)$",
    re.IGNORECASE,
)


def _is_confirm(msg: str) -> bool:
    """Check if message is a short confirmation."""
    msg = msg.strip().lower()
    if _has_cancel(msg):
        return False
    return bool(_CONFIRM_MESSAGE_RE.fullmatch(msg))


def _has_cancel(msg: str) -> bool:
    """Check if message is an explicit cancellation reply."""
    msg = msg.strip().lower()
    return bool(_CANCEL_MESSAGE_RE.fullmatch(msg))


def _should_augment_simple_word_query(message_text: str, response: str) -> bool:
    """Skip query augmentation for confirmations and action-result replies."""
    text = message_text.strip()
    if not text:
        return False
    if _is_confirm(text) or _has_cancel(text):
        return False

    response_text = response.strip()
    action_markers = (
        "加入草稿",
        "当前草稿",
        "发送「提交」",
        "发送“提交”",
        "diff Phrase",
        "草稿地址：",
        "✅ 已将",
        "✅ 已写入草稿",
        "插入编码",
        "调整到编码",
    )
    return not any(marker in response_text for marker in action_markers)


def _parse_pending_add_word(response: str) -> Optional[PendingAddWord]:
    """Parse AI response for the candidate code confirmation pattern.

    Looks for: 是否以编码 XXX 将「YYY」加入草稿
    and the numbered candidate list.
    """
    confirm_match = re.search(r'以编码\s*([a-z]+)\s*将「(.+?)」加入草稿', response)
    if not confirm_match:
        return None
    recommended_code = confirm_match.group(1)
    word = confirm_match.group(2)

    candidates: List[Tuple[str, bool]] = []
    occupied_words: Dict[str, List[str]] = {}
    for m in re.finditer(r'\d+\.\s*([a-z]+)\s*[-—–]\s*(.+)', response):
        code = m.group(1)
        desc = m.group(2)
        candidates.append((code, '已有' in desc))
        occupied_match = re.search(r'已有「(.+?)」', desc)
        if occupied_match:
            occupied_words[code] = [
                part.strip()
                for part in occupied_match.group(1).split('、')
                if part.strip()
            ]

    if not candidates:
        candidates = [(recommended_code, False)]

    return PendingAddWord(
        word=word,
        recommended_code=recommended_code,
        candidates=candidates,
        occupied_words=occupied_words,
    )


def _get_latest_assistant_message(history: Optional[List[Dict]]) -> str:
    """Return the most recent assistant message, if any."""
    if not history:
        return ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "") or "")
    return ""


def _looks_like_submit_reconfirm_prompt(response: str) -> bool:
    """Detect a prior assistant message asking the user to reconfirm submission."""
    text = (response or "").strip()
    if not text or "提交" not in text or "加入草稿" in text:
        return False

    hints = (
        "是否继续提交",
        "确认提交",
        "继续提交吗",
        "继续提审",
        "确认后继续提交",
        "回复「确认」继续提交",
        "回复“确认”继续提交",
        "确认继续提交",
    )
    return any(hint in text for hint in hints)


def _recover_pending_state_from_history(history: Optional[List[Dict]]) -> PendingState:
    """Best-effort recovery when in-memory pending state was lost."""
    assistant_message = _get_latest_assistant_message(history)
    if not assistant_message:
        return None

    pending_add = _parse_pending_add_word(assistant_message)
    if pending_add is not None:
        return pending_add

    if _looks_like_submit_reconfirm_prompt(assistant_message):
        return PendingToolConfirm(function_name="keytao_submit_batch", args={})

    return None


def _ensure_pending_add_word_guidance(response: str) -> str:
    """Append deterministic guidance for occupied candidate choices."""
    guidance = "若所选编号显示“已有…”，直接回复该编号表示添加重码；回复“编号 重新编码”或“原词 重新编码”则挪开原词。"
    if "重新编码" in response and "添加重码" in response:
        return response

    # Robust fallback: if the visible reply already contains numbered-choice wording
    # and at least one occupied slot, append guidance even when regex parsing misses.
    if "也可回复编号选其他编码" in response and "已有「" in response:
        logger.info("🧭 Appending occupied-choice guidance via fallback matcher")
        return response.rstrip() + f"\n{guidance}"

    pending = _parse_pending_add_word(response)
    if pending is None:
        return response
    if not any(occupied for _, occupied in pending.candidates):
        return response
    logger.info("🧭 Appending occupied-choice guidance via parsed pending-add matcher")
    return response.rstrip() + f"\n{guidance}"


def _build_existing_word_priority_note(word: str, lookup_entry: Dict, encode_data: Dict) -> Optional[str]:
    """Explain why an existing word uses its current code and where it ranks there."""
    phrases = lookup_entry.get("phrases", [])
    if not phrases:
        return None

    candidate_statuses = encode_data.get("candidateStatuses", [])
    candidate_index = {
        item.get("code", ""): idx
        for idx, item in enumerate(candidate_statuses)
        if isinstance(item, dict) and item.get("code")
    }

    notes: List[str] = []
    for phrase in phrases:
        code = phrase.get("code", "")
        if not code:
            continue

        idx = candidate_index.get(code)
        if idx is not None and idx > 0:
            prior_statuses = [
                item for item in candidate_statuses[:idx]
                if isinstance(item, dict) and item.get("occupied")
            ]
            if prior_statuses:
                prior_text = "；".join(
                    f"{item.get('code', '')} {item.get('label', '')}"
                    for item in prior_statuses[:3]
                )
                notes.append(f"{word} 当前用 {code}，因为更前面的候选码位已被占用：{prior_text}。")

        dup = phrase.get("duplicate_info")
        if isinstance(dup, dict) and len(dup.get("all_words", [])) > 1:
            position_label = dup.get("position_label") or "首位"
            all_words = dup.get("all_words", [])
            dup_text = "、".join(
                f"{item.get('word', '')}{f'（{item.get('label')}）' if item.get('label') else ''}"
                for item in all_words[:5]
                if item.get("word")
            )
            notes.append(f"{code} 这个码位里，{word} 排在{position_label}；同码词有：{dup_text}。")

    if not notes:
        return None
    return "\n".join(f"• {note}" for note in notes)


def _extract_prior_occupied_candidates(current_code: str, encode_data: Dict) -> List[Dict]:
    """Return occupied candidate slots before the current code."""
    candidate_statuses = encode_data.get("candidateStatuses", [])
    if not isinstance(candidate_statuses, list):
        return []
    current_index = next(
        (idx for idx, item in enumerate(candidate_statuses) if isinstance(item, dict) and item.get("code") == current_code),
        None,
    )
    if current_index is None or current_index <= 0:
        return []
    result = []
    for item in candidate_statuses[:current_index]:
        if not isinstance(item, dict) or not item.get("occupied"):
            continue
        result.append({
            "code": item.get("code", ""),
            "label": item.get("label", ""),
        })
    return result


def _extract_words_from_candidate_label(label: str) -> List[str]:
    """Extract occupied words from candidate label like 已有「甲、乙」."""
    if not label:
        return []
    match = re.search(r'已有「(.+?)」', label)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split('、') if part.strip()]


async def _generate_usage_comparison_note(
    word: str,
    current_code: str,
    prior_occupied: List[Dict],
) -> Optional[str]:
    """Ask the model for a concise common-usage comparison note."""
    if not prior_occupied or not OPENAI_API_KEY or not AsyncOpenAI:
        return None

    occupied_text = "；".join(
        f"{item.get('code', '')} {item.get('label', '')}"
        for item in prior_occupied
        if item.get("code")
    )
    occupied_words = []
    for item in prior_occupied:
        occupied_words.extend(_extract_words_from_candidate_label(str(item.get("label", ""))))
    if not occupied_text:
        return None

    try:
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=min(OPENAI_TIMEOUT, 30.0),
        )
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.3,
            max_tokens=180,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是中文输入法助手。请用1到2句简短中文，比较当前词和前面占位词在日常使用中的常见场景/常用度差异。"
                        "语气克制，不要绝对化，不要使用项目符号。"
                        "优先直接点名占位词，并明确这只是日常语感层面的比较，不等于实际码序规则。"
                        "最后顺带点明：当前码位顺序仍以现有词库占位为准。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"当前词：{word}\n"
                        f"当前编码：{current_code}\n"
                        f"更前面被占用的候选码位：{occupied_text}\n"
                        f"前面占位词：{'、'.join(occupied_words) if occupied_words else '未知'}"
                    ),
                },
            ],
        )
        if not response.choices:
            return None
        content = (response.choices[0].message.content or "").strip()
        return content or None
    except Exception as error:
        logger.warning(f"Failed to generate usage comparison note for {word}: {error}")
        return None


async def _augment_simple_word_query_response(
    message_text: str,
    response: str,
    platform: str,
    user_id: str,
) -> str:
    """Append deterministic code-priority notes for simple word-only queries."""
    if not _should_augment_simple_word_query(message_text, response):
        return response

    words = _extract_pure_chinese_words(message_text)
    if not words:
        return response

    lookup_json = await call_tool_function(
        "keytao_lookup_by_words_batch", {"words": words}, platform, user_id,
    )
    try:
        lookup_data = json.loads(lookup_json)
    except Exception:
        return response
    if not lookup_data.get("success"):
        return response

    lookup_map = {
        item.get("word", ""): item
        for item in lookup_data.get("results", [])
        if isinstance(item, dict) and item.get("word")
    }

    note_blocks: List[str] = []
    for word in words:
        lookup_entry = lookup_map.get(word, {})
        if not lookup_entry.get("phrases"):
            continue
        encode_json = await call_tool_function(
            "keytao_encode", {"word": word}, platform, user_id,
        )
        try:
            encode_data = json.loads(encode_json)
        except Exception:
            continue
        note = _build_existing_word_priority_note(word, lookup_entry, encode_data)
        note_lines = []
        if note:
            note_lines = [
                line for line in note.splitlines()
                if line.strip() and line.strip() not in response
            ]
        comparison_notes: List[str] = []
        for phrase in lookup_entry.get("phrases", []):
            code = phrase.get("code", "")
            if not code:
                continue
            prior_occupied = _extract_prior_occupied_candidates(code, encode_data)
            comparison = await _generate_usage_comparison_note(word, code, prior_occupied)
            comparison_line = f"• 常用度对比：{comparison}" if comparison else ""
            if comparison_line and comparison_line not in response:
                comparison_notes.append(f"• 常用度对比：{comparison}")
        if note_lines or comparison_notes:
            block_parts = [f"{word} 的编码位置说明："]
            if note_lines:
                block_parts.extend(note_lines)
            if comparison_notes:
                block_parts.extend(comparison_notes)
            note_blocks.append("\n".join(block_parts))

    if not note_blocks:
        return response
    return response.rstrip() + "\n\n补充说明：\n" + "\n\n".join(note_blocks)


# ---------------------------------------------------------------------------
# Platform detection & OneBot helpers
# ---------------------------------------------------------------------------

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


def extract_onebot_reply_id(event: Event) -> Optional[str]:
    """Extract replied message id from OneBot v11 message segments."""
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
    except Exception:
        pass
    return ''.join(parts).strip()


async def build_reply_context(bot: Bot, event: Event) -> str:
    """Build reply context for Telegram and OneBot v11."""
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
    except ImportError:
        TelegramBot = None
    try:
        from nonebot.adapters.onebot.v11 import Bot as QQBot
    except ImportError:
        QQBot = None

    # --- Telegram ---
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
        reply_text = getattr(reply_to_message, 'text', None)
        if reply_from and reply_text:
            reply_from_id = getattr(reply_from, 'id', None)
            reply_from_name = getattr(reply_from, 'first_name', '未知用户')
            if bot_id and reply_from_id == bot_id:
                return (
                    f"\n\n【用户正在回复你的消息】\n被引用的消息内容：\n{reply_text}\n\n"
                    "⚠️ 用户的回复是针对这条消息的，请根据这条消息的内容理解用户意图。"
                )
            return (
                f"\n\n【用户正在回复其他人的消息】\n被引用消息的发送者：{reply_from_name}\n"
                f"被引用的消息内容：\n{reply_text}\n\n"
                "⚠️ 用户回复的不是你的消息，如果用户说的是操作指令（如'是'、'确认'、'提交'），"
                "应该提醒用户：你需要回复bot的消息才能确认操作。"
            )
        return ""

    # --- OneBot v11 (QQ) ---
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
        reply_text = extract_onebot_plaintext(
            reply_payload.get('message') if isinstance(reply_payload, dict) else None
        )
        if not reply_text and isinstance(reply_payload, dict):
            reply_text = str(reply_payload.get('raw_message', '')).strip()
        if not reply_text:
            return ""

        bot_id = str(getattr(bot, 'self_id', ''))
        if bot_id and reply_from_id == bot_id:
            return (
                f"\n\n【用户正在回复你的消息】\n被引用的消息内容：\n{reply_text}\n\n"
                "⚠️ 用户的回复是针对这条消息的，请根据这条消息的内容理解用户意图。"
            )
        return (
            f"\n\n【用户正在回复其他人的消息】\n被引用消息的发送者：{reply_from_name}\n"
            f"被引用的消息内容：\n{reply_text}\n\n"
            "⚠️ 用户回复的不是你的消息，如果用户说的是操作指令（如'是'、'确认'、'提交'），"
            "应该提醒用户：你需要回复bot的消息才能确认操作。"
        )

    return ""


# ---------------------------------------------------------------------------
# Cross-platform message handling rule
# ---------------------------------------------------------------------------

async def should_handle(bot: Bot, event: Event) -> bool:
    """
    Custom rule:
    - QQ: to_me() or trigger keywords
    - Telegram: private always, group when mentioned/replied
    """
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.telegram.event import (
            PrivateMessageEvent,
            GroupMessageEvent,
        )
        from nonebot.adapters.onebot.v11 import Bot as QQBot
        from nonebot.adapters.onebot.v11.event import (
            PrivateMessageEvent as QQPrivateMessageEvent,
            GroupMessageEvent as QQGroupMessageEvent,
        )

        if isinstance(bot, TelegramBot):
            if isinstance(event, PrivateMessageEvent):
                return True
            if isinstance(event, GroupMessageEvent):
                reply_to_message = getattr(event, 'reply_to_message', None)
                if reply_to_message:
                    bot_info = await bot.get_me()
                    reply_from = getattr(reply_to_message, 'from_', None)
                    if reply_from and reply_from.id == bot_info.id:
                        return True

                message_text = event.get_plaintext().strip()
                bot_info = await bot.get_me()
                bot_username = bot_info.username

                try:
                    message_to_check = getattr(event, 'original_message', event.message)
                    for segment in message_to_check:
                        if segment.type == 'mention':
                            mention_text = segment.data.get('text', '')
                            if mention_text == f"@{bot_username}":
                                return True
                except Exception:
                    pass

                if (GROUP_TRIGGER_KEYWORD_ANY in message_text
                        or message_text.startswith(GROUP_TRIGGER_KEYWORD_START)):
                    return True
                return False
            return False

        elif isinstance(bot, QQBot):
            if isinstance(event, QQPrivateMessageEvent):
                return True
            if isinstance(event, QQGroupMessageEvent):
                if await to_me()(bot, event, {}):
                    return True
                message_text = event.get_plaintext().strip()
                if (GROUP_TRIGGER_KEYWORD_ANY in message_text
                        or message_text.startswith(GROUP_TRIGGER_KEYWORD_START)):
                    return True
                return False
            return await to_me()(bot, event, {})

        else:
            return await to_me()(bot, event, {})

    except Exception as e:
        logger.error(f"Error in should_handle rule: {e}")
        return False


# ---------------------------------------------------------------------------
# Conversation key / history helpers
# ---------------------------------------------------------------------------

def get_conversation_key(bot: Bot, event: Event) -> Tuple[str, str]:
    return extract_platform_info(bot, event)


def get_history(key: Tuple[str, str]) -> List[Dict]:
    platform, user_id = key
    return history_store.get_history(platform, user_id, limit=MAX_HISTORY_MESSAGES)


def add_to_history(key: Tuple[str, str], user_message: str, assistant_message: str):
    platform, user_id = key
    history_store.add_conversation_round(platform, user_id, user_message, assistant_message)


def clear_history(key: Tuple[str, str]):
    platform, user_id = key
    history_store.clear_history(platform, user_id)


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------

_INJECT_PLATFORM_TOOLS = frozenset({
    'keytao_create_phrase', 'keytao_submit_batch',
    'keytao_list_draft_items', 'keytao_remove_draft_item',
    'keytao_batch_add_to_draft', 'keytao_batch_remove_draft_items',
    'keytao_shift_phrase_code', 'keytao_recall_batch', 'keytao_get_batch_preview',
})
tool_executor = ToolExecutor(skills_manager.get_tool_function, _INJECT_PLATFORM_TOOLS)


async def call_tool_function(
    tool_name: str,
    arguments: Dict,
    platform: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Call a tool function and return result as JSON string."""
    return await tool_executor.call(tool_name, arguments, ToolContext(platform, user_id))


# ---------------------------------------------------------------------------
# Direct execution helpers (bypasses AI for simple confirmations)
# ---------------------------------------------------------------------------

async def _execute_add_to_draft(
    word: str, code: str, platform: str, user_id: str,
) -> str:
    """Directly add a word to draft and return formatted response."""
    result_json = await call_tool_function(
        "keytao_create_phrase", {"word": word, "code": code}, platform, user_id,
    )
    data = json.loads(result_json)

    if data.get("not_bound"):
        return _BIND_HELP_TEXT

    if data.get("requiresConfirmation"):
        conv_key = (platform, user_id)
        conversation_state_store.set(conv_key, PendingToolConfirm(
            function_name="keytao_create_phrase",
            args={"word": word, "code": code},
        ))
        warnings = data.get("warnings", [])
        warn_text = "\n".join(
            f"⚠️ {w.get('message', w) if isinstance(w, dict) else w}"
            for w in warnings
        ) if warnings else data.get("message", "存在重码警告")
        return f"{warn_text}\n\n确认添加吗？回复「确认」继续，「取消」放弃。"

    if not data.get("success"):
        return f"添加失败：{data.get('message', '未知错误')} qwq"

    header = f"✅ 已将「{word}」以编码 {code} 加入草稿\n"
    return header + await _format_draft_response(data, platform, user_id)


async def _execute_shift_to_code(
    word: str, target_code: str, platform: str, user_id: str,
) -> str:
    """Insert/move a word into an occupied code and shift occupants forward."""
    result_json = await call_tool_function(
        "keytao_shift_phrase_code", {"word": word, "target_code": target_code}, platform, user_id,
    )
    data = json.loads(result_json)

    if data.get("not_bound"):
        return _BIND_HELP_TEXT

    if not data.get("success"):
        return f"调整编码失败：{data.get('message', '未知错误')} qwq"

    shifted = data.get("shiftPlan", {}).get("shifted", [])
    if shifted:
        header = f"✅ 已将「{word}」插入编码 {target_code}，并顺延 {len(shifted)} 条\n"
    else:
        header = f"✅ 已将「{word}」调整到编码 {target_code}\n"
    return header + await _format_draft_response(data, platform, user_id)


def _resolve_shift_target_code(state: PendingAddWord, msg: str) -> Optional[str]:
    """Resolve which occupied candidate the user wants to shift for."""
    if "重新编码" not in msg:
        return None

    digit_match = re.search(r'(\d+)', msg)
    if digit_match:
        idx = int(digit_match.group(1)) - 1
        if 0 <= idx < len(state.candidates):
            code, occupied = state.candidates[idx]
            if occupied:
                return code

    for code, occupied in state.candidates:
        if not occupied:
            continue
        for occupant_word in state.occupied_words.get(code, []):
            if occupant_word and occupant_word in msg:
                return code

    occupied_codes = [code for code, occupied in state.candidates if occupied]
    if len(occupied_codes) == 1:
        return occupied_codes[0]
    return None


async def _execute_confirmed_tool(
    state: PendingToolConfirm, platform: str, user_id: str,
) -> str:
    """Re-call a tool with confirmed=True and return formatted response."""
    args = {**state.args, "confirmed": True}
    result_json = await call_tool_function(state.function_name, args, platform, user_id)
    data = json.loads(result_json)

    if state.function_name == "keytao_submit_batch":
        if data.get("success"):
            batch_url = data.get("batchUrl", "")
            pr_url = data.get("prUrl", "")
            parts = ["✅ 草稿已成功提交审核！"]
            if batch_url:
                parts.append(f"\n草稿地址：{batch_url}")
            if pr_url:
                parts.append(f"PR：{pr_url}")
            return "\n".join(parts)
        return f"提交失败：{data.get('message', '未知错误')} qwq"

    if data.get("success"):
        header = "✅ 已确认添加到草稿\n"
        return header + await _format_draft_response(data, platform, user_id)
    return f"操作失败：{data.get('message', '未知错误')} qwq"


async def _format_draft_response(data: Dict, platform: str, user_id: str) -> str:
    """Format draft state (summary + diff + items + URL) after an operation."""
    preview_json = await call_tool_function("keytao_get_batch_preview", {}, platform, user_id)
    preview = json.loads(preview_json)

    snapshot = data.get("draft_snapshot")
    if not snapshot:
        list_json = await call_tool_function("keytao_list_draft_items", {}, platform, user_id)
        list_data = json.loads(list_json)
        if list_data.get("success"):
            snapshot = {
                "count": list_data.get("count", 0),
                "items": list_data.get("items", []),
                "summary": list_data.get("summary", {}),
            }

    parts: List[str] = []

    # Notes from Delete operations
    for note in data.get("notes", []):
        nw = note.get("word", "")
        nc = note.get("code", "")
        nt = note.get("type", "")
        type_label = {"Phrase": "词组", "Single": "单字"}.get(nt, nt)
        parts.append(f"📝 注意：{nw}（{nc}，{type_label}）已从词库标记删除")

    # Summary line
    summary = None
    if snapshot:
        summary = snapshot.get("summary")
    if not summary and preview.get("success"):
        summary = preview.get("summary")
    if summary:
        parts.append(
            f"+{summary.get('added', 0)} 新增  "
            f"~{summary.get('modified', 0)} 修改  "
            f"-{summary.get('deleted', 0)} 删除"
        )

    # Diff block
    diff_text = preview.get("diff_text", "") if preview.get("success") else ""
    if diff_text:
        parts.append(f"\n{diff_text}")

    # Draft items
    if snapshot:
        items = snapshot.get("items", [])
        count = snapshot.get("count", len(items))
        parts.append(f"\n当前草稿（共 {count} 条）：")
        for item in items:
            action_label = item.get("action_label") or {
                "Create": "新增", "Change": "修改", "Delete": "删除",
            }.get(item.get("action", ""), "")
            display = item.get("display_label") or f"{item.get('word', '')} → {item.get('code', '')}"
            parts.append(f"• {action_label} {display}")

    # Batch URL
    batch_url = data.get("batchUrl") or preview.get("batchUrl", "")
    if batch_url:
        parts.append(f"\n草稿地址：{batch_url}")

    parts.append("\n发送「提交」以提交该草稿")
    return "\n".join(parts)


async def _handle_pending_add_word(
    state: PendingAddWord,
    message: str,
    platform: str,
    user_id: str,
    history: List[Dict],
) -> Optional[str]:
    """Handle user response to a pending add-word prompt.

    Returns a response string if handled directly, None to fall through to AI.
    """
    msg = message.strip()
    shift_target_code = _resolve_shift_target_code(state, msg)
    if shift_target_code is not None:
        return await _execute_shift_to_code(state.word, shift_target_code, platform, user_id)

    target_code: Optional[str] = None
    is_occupied = False

    # Numeric choice (e.g. "1", "2", "3")
    if msg.isdigit():
        idx = int(msg) - 1
        if 0 <= idx < len(state.candidates):
            target_code, is_occupied = state.candidates[idx]
        else:
            conv_key = (platform, user_id)
            conversation_state_store.set(conv_key, state)
            return f"请选择 1-{len(state.candidates)} 之间的编号 owo"

    # Simple confirmation -> use recommended code
    elif _is_confirm(msg):
        target_code = state.recommended_code
        for c, occ in state.candidates:
            if c == target_code:
                is_occupied = occ
                break

    if target_code is None:
        return None  # unrecognized input, let AI handle as new request

    # Empty slot -> direct execution (no AI needed)
    if not is_occupied:
        return await _execute_add_to_draft(state.word, target_code, platform, user_id)

    return await _execute_confirmed_tool(
        PendingToolConfirm(
            function_name="keytao_create_phrase",
            args={"word": state.word, "code": target_code},
        ),
        platform,
        user_id,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CORE = """你是键道输入法的AI助手"喵喵"。
你像一个聪明、自然、反应快的人类助手一样说话：该聊天时聊天，该办事时办事，该调用工具时果断调用。

━━━━━━━━━━━━━━━━━━━━━
核心原则
━━━━━━━━━━━━━━━━━━━━━

1. 消息处理
   • 只处理标有 [当前请求] 的消息
   • 带时间标签 [Xm ago] 的是历史记录，不要重复处理
   • 带 [系统提示] 标签的指令必须严格执行

2. 必须调用工具（绝不凭记忆回答编码问题）
   • 查词/编码 → 调用查询工具
   • 文档/规则 → 调用文档工具
   • 增删改词条 → 调用草稿工具

2.1 草稿编辑安全红线
    • 用户说"把 A 改到 xxx"且 xxx 已被占用时，可以顺延插入位置及后续词
    • 顺延必须调用 keytao_shift_phrase_code(word=A, target_code=xxx)，禁止手工计算
    • 被挤走的 B 必须用 B 自己的 keytao_encode 候选编码链找下一位，不能沿用 A 的编码链
    • 每次顺延都必须确认目标码为空，或继续顺延该目标码上的词；无法继续时停止并告知用户
    • 回复必须说明顺延计算了哪些词，例如：换言之 hyfio→hyfioo
    • 禁止先批量删除大量草稿条目再按模型规划重建，除非用户明确要求清空/批量删除

3. 查词完整流程（严格遵循，不得省略！）
   触发：用户查词、问怎么打、想加词

   【特殊默认规则】如果用户只发了一个或多个中文词/短词（例如“增香”“卧龙凤雏”或“增香 卧龙凤雏”），
     默认视为：既想知道这些词的大致词义，也想知道它们在键道里的编码/候选/排序信息。
     必须主动进入查词流程，不要只闲聊或只回一句“这是个词”。
     词义解释可以直接用你的语言能力简短说明，不必额外查外部资料；
     但编码、候选码、重码顺序必须来自工具结果，不能凭空编造。
     多个词时优先使用批量查询工具，并按词逐个整理结果。

   【第一步】同时调用：
     keytao_encode(word) + keytao_lookup_by_word(word)
         如果用户指定了目标编码/编码系列（例如“放到 ffb 系列”“用 ff=zh,zh”），
         必须调用 keytao_encode(word, requested_code=目标编码或系列前缀)，用 requestedCodeAnalysis 判断是否支持。

   【第二步】判断：
     A) 词库已有 → 展示词库位置 + 拆分，流程结束
     B) 词库没有 → 必须继续第三步

   【第三步】查候选编码占用情况：
         优先使用 keytao_encode 返回的 candidateStatuses（已查占用）。
         如果 occupancyChecked=false 或没有 candidateStatuses，才取 candidateCodes/codes + altCodes，
         调用 keytao_lookup_by_codes_batch 查每个码位。
         飞键候选必须以工具返回的 altCodes / flyKeyVariants / candidateStatuses 为准；
         支持固定规则组合候选，如 zh 的 q/f 双键位组合，禁止自己泛化到规则外键位。
         ⚠️ 禁止向用户展示“待查占用”；回复前必须得到“已有「...」”或“空位”。

   【第四步】展示拆分 + 候选编码列表，格式：

     「词」（N字词）的拆分和候选编码：

     逐字拆分：
     • 字（pin）音码 XX　字根 ...　形码 ...

     候选编码：
     1. abcd — 已有「旧词」
     2. abcde — ✅ 推荐（空位）
     3. abcdea — 空位

         是否以编码 abcde 将「词」加入草稿？也可回复编号选其他编码。
         若所选编号显示“已有…”，直接回复该编号表示添加重码；回复“编号 重新编码”或“原词 重新编码”则挪开原词。

   ⚠️ 确认句格式必须固定：「是否以编码 XXX 将「YYY」加入草稿」——系统靠此提取上下文
     ⚠️ 推荐编码使用 keytao_encode.recommendedCode；若 candidateStatuses 中有 ✅ 推荐，以该空位为准
     ⚠️ 禁止只说"未收录"就结束，必须给出可操作的加词建议
     ⚠️ 这一步只展示不写入！用户确认后由系统自动处理

   【用户只发一个或多个词时的回复要求】
     • 每个词都先用 1-2 句解释它的大致含义/常见用法
     • 如果 keytao_lookup_by_word / keytao_lookup_by_words_batch 命中词库：
       1. 说明该词已有编码
       2. 如果该编码存在 duplicate_info / all_words，主动说明该词在同码词里的排序位置
       3. 可以顺带列出同码的前后相关词，但只限工具结果里真实存在的词
     • 如果词库没有该词：
       1. 给出简短词义
       2. 再给拆分、候选编码和加词引导
     • 多个词时按词分段回答，避免把多个词混在一段里
     • 不要把“相关词”发散成大段百科，只需围绕当前词和工具查到的同码词/占位词简洁说明

4. 提交草稿
   • 仅当用户明确说"提交/提审/发起审核"时调 keytao_submit_batch
   • "确认/好/是"不是提交指令
   • 提交成功后不再调用任何其他工具

5. 查看草稿
   • 同时调用 keytao_get_batch_preview 和 keytao_list_draft_items
   • 按草稿 SKILL 文档中的格式合并展示

6. Delete 操作的 notes
   成功响应含 notes 字段时，必须把 notes 内容告知用户

7. 聊天判断
   • 闲聊/问候/倾诉/玩笑 → 自然回复，不调工具
   • 查词/编码/规则/加词 → 调工具
   • 结合上下文判断，短消息不等于查词也不等于闲聊

8. 格式规则
   • 所有平台输出完整 URL，禁止用占位符替代
   • 使用纯文本格式（不要 Markdown）
   • 工具只能通过 API tool_calls 调用，绝不在文本中手写

━━━━━━━━━━━━━━━━━━━━━
回复风格
━━━━━━━━━━━━━━━━━━━━━

• 温暖自然，简洁直接
• 可以适度活泼，不要堆表情
• 不同信息分段，空行隔开
"""


# ---------------------------------------------------------------------------
# Structural message preprocessor (bypasses AI for well-defined batch ops)
# ---------------------------------------------------------------------------

_RE_REPLACE_CHAR = re.compile(
    r'将(?:这些|这批|以下|下列)?(?:\S{0,12}?)(?:词条|词|字)?(?:中|中的|里|里的|的)?'
    r'["\u201c\u300c]?(.)["\u201d\u300d]?改[为成]["\u201c\u300c]?(.)["\u201d\u300d]?[：:，,\s]'
)
_RE_WORD_CODE_LINE = re.compile(r'^(\S+)\s+([a-z]+)\s*$')
_TYPE_HINTS = [
    ("声笔笔单字", "CSSSingle"),
    ("CSSSingle", "CSSSingle"),
    ("css-single", "CSSSingle"),
    ("声笔笔", "CSS"),
    ("CSS", "CSS"),
    ("词组", "Phrase"),
    ("词语", "Phrase"),
    ("单字", "Single"),
    ("补充", "Supplement"),
    ("符号", "Symbol"),
    ("链接", "Link"),
    ("英文", "English"),
]


def _extract_explicit_phrase_type(message: str) -> Optional[str]:
    for hint, phrase_type in _TYPE_HINTS:
        if hint in message:
            return phrase_type
    return None


async def _try_handle_replace_char(
    message: str, platform: str, user_id: str
) -> Optional[str]:
    """Detect '将X改成Y + word-code list' pattern and handle directly in Python."""
    m = _RE_REPLACE_CHAR.search(message)
    if not m:
        return None

    old_char, new_char = m.group(1), m.group(2)
    phrase_type = _extract_explicit_phrase_type(message)
    items = []
    for line in message.splitlines():
        lm = _RE_WORD_CODE_LINE.match(line.strip())
        if not lm:
            continue
        old_word, code = lm.group(1), lm.group(2)
        if old_char not in old_word:
            continue
        new_word = old_word.replace(old_char, new_char)
        item = {"action": "Change", "old_word": old_word, "word": new_word, "code": code}
        if phrase_type:
            item["type"] = phrase_type
        items.append(item)

    if not items:
        return None

    logger.info(f"[replace_char] Detected pattern '{old_char}'→'{new_char}', {len(items)} items, bypassing AI")
    result_str = await call_tool_function("keytao_batch_add_to_draft", {"items": items}, platform, user_id)
    try:
        data = json.loads(result_str)
    except Exception:
        return "呜呜，批量修改失败 qwq"

    success = data.get("successCount", 0)
    failed = data.get("failedCount", 0)
    skipped = data.get("skippedCount", 0)

    parts = [f"✅ 已将 {len(items)} 个词中的「{old_char}」替换为「{new_char}」"]
    parts.append(f"成功 {success} 条" + (f"，跳过 {skipped} 条" if skipped else "") + (f"，失败 {failed} 条" if failed else ""))

    if data.get("failed"):
        failed_lines = [f"  • {f['word']}（{f['code']}）：{f['reason']}" for f in data["failed"][:5]]
        parts.append("❌ 未写入：\n" + "\n".join(failed_lines))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core AI response function (platform-agnostic)
# ---------------------------------------------------------------------------

async def get_ai_response_core(
    message: str,
    platform: str,
    user_id: str,
    history: Optional[List[Dict]] = None,
    reply_context: str = "",
    max_iterations: int = 20,
) -> Optional[str]:
    """Call OpenAI-compatible API with function calling support.

    Platform-agnostic: works for QQ, Telegram, and web API calls.
    """
    if not OPENAI_API_KEY or not AsyncOpenAI:
        return "❌ AI 服务未配置，请联系管理员"

    try:
        client_cls = AsyncOpenAI
        runtime = AgentRuntimeConfig(
            model=OPENAI_MODEL,
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=OPENAI_TEMPERATURE,
            timeout=OPENAI_TIMEOUT,
        )
        orchestrator = AgentOrchestrator(
            client_factory=lambda: client_cls(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                timeout=OPENAI_TIMEOUT,
            ),
            runtime=runtime,
            skills_manager=skills_manager,
            tool_executor=tool_executor,
            state_store=conversation_state_store,
            bind_help_text=_BIND_HELP_TEXT,
            system_prompt_core=SYSTEM_PROMPT_CORE,
        )
        return await orchestrator.run(
            message=message,
            context=AgentRequestContext(
                platform=platform,
                user_id=user_id,
                history=history,
                reply_context=reply_context,
            ),
            max_iterations=max_iterations,
        )

    except Exception as e:
        logger.error(f"API error: {e}")
        return "呜呜，AI 服务暂时不可用 qwq 等等再来找我吧～"


async def get_openai_response(
    message: str,
    bot: Bot,
    event: Event,
    history: Optional[List[Dict]] = None,
    max_iterations: int = 20,
) -> Optional[str]:
    """NoneBot wrapper: extract platform context then call get_ai_response_core."""
    platform, user_id = extract_platform_info(bot, event)
    reply_context = await build_reply_context(bot, event)
    return await get_ai_response_core(
        message, platform, user_id, history, reply_context, max_iterations,
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

clear_cmd = on_command(
    "clear", aliases={"清空对话", "清空历史"},
    rule=Rule(should_handle), priority=5, block=True,
)


async def should_handle_clear_message(bot: Bot, event: Event) -> bool:
    if not _is_clear_command_text(event.get_plaintext()):
        return False
    return await should_handle(bot, event)


clear_message = on_message(
    rule=Rule(should_handle_clear_message),
    priority=4,
    block=True,
)


@clear_cmd.handle()
async def handle_clear(bot: Bot, event: Event):
    await _handle_clear(bot, event, clear_cmd)


@clear_message.handle()
async def handle_clear_message(bot: Bot, event: Event):
    await _handle_clear(bot, event, clear_message)


async def _handle_clear(bot: Bot, event: Event, matcher):
    conv_key = get_conversation_key(bot, event)
    clear_history(conv_key)
    conversation_state_store.delete(conv_key)
    await matcher.finish("好哒～ 对话历史已清空！我们重新开始吧 owo")


ai_chat = on_message(rule=should_handle, priority=99, block=True)


@ai_chat.handle()
async def handle_ai_chat(bot: Bot, event: Event):
    # Platform-specific imports (may not all be installed)
    try:
        from nonebot.adapters.onebot.v11 import MessageSegment as QQMessageSegment
    except ImportError:
        QQMessageSegment = None

    message_text = event.get_plaintext().strip()
    if not message_text:
        await ai_chat.finish("你好呀～ owo 我是喵喵，键道输入法的助手！有什么可以帮你的吗？")
        return
    normalized_message_text = _strip_command_message_prefixes(message_text) or message_text

    platform, user_id = extract_platform_info(bot, event)
    conv_key = (platform, user_id)
    response: Optional[str] = None
    history: Optional[List[Dict]] = None

    # ===== Phase 1: Check pending state =====
    state = conversation_state_store.pop(conv_key)
    if state is None:
        history = get_history(conv_key)
        state = _recover_pending_state_from_history(history)
        if state is not None:
            logger.info(
                "♻️ Recovered pending state from history: "
                f"{state.__class__.__name__} for {platform}:{user_id}"
            )

    if state is not None:
        if _has_cancel(normalized_message_text):
            response = "好的，已取消 owo"

        elif isinstance(state, PendingAddWord):
            if history is None:
                history = get_history(conv_key)
            response = await _handle_pending_add_word(
                state, normalized_message_text, platform, user_id, history,
            )
            # response is None → unrecognized input, fall through to Phase 2

        elif isinstance(state, PendingToolConfirm):
            # "提交" when pending state is submit_batch also counts as confirm
            is_submit_reconfirm = (
                state.function_name == "keytao_submit_batch"
                and normalized_message_text.strip() in {"提交", "提审"}
            )
            if _is_confirm(normalized_message_text) or is_submit_reconfirm:
                response = await _execute_confirmed_tool(state, platform, user_id)
            # else: response stays None, fall through to AI as new request

    # ===== Phase 2: AI response (if not handled directly) =====
    if response is None:
        if history is None:
            history = get_history(conv_key)
        reply_context = await build_reply_context(bot, event)
        response = await get_ai_response_core(
            normalized_message_text, platform, user_id, history, reply_context,
        )

    if not response:
        await ai_chat.finish("呜呜，处理请求时出错了 qwq 要不再试一次？")
        return

    response = _ensure_pending_add_word_guidance(response)
    response = await _augment_simple_word_query_response(
        normalized_message_text, response, platform, user_id,
    )

    # ===== Phase 3: Detect new pending state from AI response =====
    if not conversation_state_store.contains(conv_key):
        pending = _parse_pending_add_word(response)
        if pending:
            conversation_state_store.set(conv_key, pending)
            logger.info(
                f"📌 Saved PendingAddWord: {pending.word}@{pending.recommended_code} "
                f"({len(pending.candidates)} candidates)"
            )

    # Save conversation history
    add_to_history(conv_key, normalized_message_text, response)

    # ===== Phase 4: Platform-specific reply =====
    bot_module = bot.__class__.__module__

    # --- Telegram ---
    if 'telegram' in bot_module.lower():
        tg_text = _to_markdownv2(response)
        message_id = getattr(event, 'message_id', None)
        if message_id:
            try:
                await bot.send(
                    event=event,
                    message=tg_text,
                    reply_to_message_id=message_id,
                    parse_mode="MarkdownV2",
                )
                return
            except Exception:
                try:
                    await bot.send(
                        event=event,
                        message=response,
                        reply_to_message_id=message_id,
                    )
                    return
                except Exception:
                    pass
        try:
            await ai_chat.finish(tg_text, parse_mode="MarkdownV2")
        except Exception:
            await ai_chat.finish(response)

    # --- QQ (OneBot v11) ---
    elif 'onebot' in bot_module.lower() or bot.__class__.__name__ == 'Bot':
        qq_text = _strip_markdown(response)
        qq_msg_id = getattr(event, 'message_id', None)
        if qq_msg_id and QQMessageSegment:
            try:
                await bot.send(
                    event=event,
                    message=QQMessageSegment.reply(qq_msg_id) + qq_text,
                )
                return
            except Exception:
                pass
        await ai_chat.finish(qq_text)

    # --- Other ---
    else:
        await ai_chat.finish(response)
