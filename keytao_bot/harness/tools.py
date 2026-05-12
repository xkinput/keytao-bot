"""Tool execution adapter for the agent harness."""
import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from nonebot.log import logger


@dataclass(frozen=True)
class ToolContext:
    platform: Optional[str] = None
    user_id: Optional[str] = None
    current_message: Optional[str] = None


_DELETE_INTENT_RE = re.compile(r"删除|删掉|移除|撤销|清空|清理|全部删|都删")
_PROTECTED_WORD_RE = r"(?:别动|不要动|别改|不要改|不动|保持)"


def _is_word_protected(message: str, word: str) -> bool:
    escaped_word = re.escape(word)
    return bool(
        re.search(escaped_word + r".{0,8}" + _PROTECTED_WORD_RE, message)
        or re.search(_PROTECTED_WORD_RE + r".{0,8}" + escaped_word, message)
    )


def _find_code_reassignments(items: object) -> List[Dict[str, str]]:
    if not isinstance(items, list):
        return []

    deletes: Dict[str, set[str]] = {}
    creates: Dict[str, set[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        code = item.get("code")
        action = item.get("action", "Create")
        if not isinstance(word, str) or not isinstance(code, str):
            continue
        if action == "Delete":
            deletes.setdefault(word, set()).add(code)
        elif action == "Create":
            creates.setdefault(word, set()).add(code)

    reassignments: List[Dict[str, str]] = []
    for word, old_codes in deletes.items():
        for new_code in creates.get(word, set()):
            for old_code in old_codes:
                if old_code != new_code:
                    reassignments.append({"word": word, "oldCode": old_code, "newCode": new_code})
    return reassignments


class ToolExecutor:
    """Executes registered skills and injects platform context when needed."""

    def __init__(self, get_tool_function: Callable[[str], Optional[Callable]], context_tools: frozenset[str]):
        self._get_tool_function = get_tool_function
        self._context_tools = context_tools

    async def call(self, tool_name: str, arguments: Dict, context: ToolContext) -> str:
        policy_error = self._validate_policy(tool_name, arguments, context)
        if policy_error:
            logger.warning(f"Tool {tool_name} blocked by policy: {policy_error}")
            return json.dumps(policy_error, ensure_ascii=False)

        tool_func = self._get_tool_function(tool_name)
        if not tool_func:
            return json.dumps({"error": f"Tool {tool_name} not found"}, ensure_ascii=False)

        call_args = dict(arguments)
        try:
            if tool_name in self._context_tools:
                if not context.platform or not context.user_id:
                    return json.dumps(
                        {"error": "内部错误：无法获取用户平台信息"}, ensure_ascii=False
                    )
                call_args["platform"] = context.platform
                call_args["platform_id"] = context.user_id

            result = await tool_func(**call_args)
            result_json = json.dumps(result, ensure_ascii=False)
            logger.info(f"Tool {tool_name} result: {result_json[:300]}")
            return result_json
        except Exception as error:
            logger.error(f"Tool {tool_name} error: {type(error).__name__}: {error}")
            return json.dumps({"error": str(error)}, ensure_ascii=False)

    def _validate_policy(self, tool_name: str, arguments: Dict, context: ToolContext) -> Optional[Dict]:
        message = context.current_message or ""
        if tool_name == "keytao_batch_remove_draft_items" and message:
            ids = arguments.get("ids")
            if isinstance(ids, list) and len(ids) > 3 and not _DELETE_INTENT_RE.search(message):
                return {
                    "success": False,
                    "policyBlocked": True,
                    "message": "安全拦截：当前消息不是批量删除请求，禁止一次删除多个草稿条目。请只删除本次明确需要替换的条目，或先向用户确认。",
                    "blockedIds": ids,
                }

        if tool_name != "keytao_batch_add_to_draft":
            return None

        reassignments = _find_code_reassignments(arguments.get("items"))
        if not reassignments or not message:
            return None

        blocked = [
            item for item in reassignments
            if item["word"] not in message or _is_word_protected(message, item["word"])
        ]
        if not blocked:
            return None

        blocked_labels = [
            f"{item['word']} {item['oldCode']}→{item['newCode']}"
            for item in blocked
        ]
        return {
            "success": False,
            "policyBlocked": True,
            "message": "安全拦截：禁止手工迁移未点名词条。需要插入已占用编码并顺延时，必须调用 keytao_shift_phrase_code，让工具按每个被挤词自己的 encode 候选链计算。",
            "blockedReassignments": blocked_labels,
        }
