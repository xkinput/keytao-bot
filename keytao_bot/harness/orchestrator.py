"""OpenAI-compatible agent/tool orchestration loop."""
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from nonebot.log import logger

from .state import MemoryConversationStateStore, PendingToolConfirm
from .tools import ToolContext, ToolExecutor


class DuplicateToolCallAbort(Exception):
    pass


@dataclass(frozen=True)
class AgentRuntimeConfig:
    model: str
    max_tokens: int
    temperature: float
    timeout: float
    max_tokens_cap: int = 8000


@dataclass(frozen=True)
class AgentRequestContext:
    platform: str
    user_id: str
    history: Optional[List[Dict]] = None
    reply_context: str = ""


class AgentOrchestrator:
    """Runs the model/tool loop and persists tool-confirmation state."""

    def __init__(
        self,
        client_factory: Callable[[], object],
        runtime: AgentRuntimeConfig,
        skills_manager: object,
        tool_executor: ToolExecutor,
        state_store: MemoryConversationStateStore,
        bind_help_text: str,
        system_prompt_core: str,
    ):
        self._client_factory = client_factory
        self._runtime = runtime
        self._skills_manager = skills_manager
        self._tool_executor = tool_executor
        self._state_store = state_store
        self._bind_help_text = bind_help_text
        self._system_prompt_core = system_prompt_core

    async def run(
        self,
        message: str,
        context: AgentRequestContext,
        max_iterations: int = 20,
    ) -> Optional[str]:
        client = self._client_factory()

        platform_label = {'telegram': 'Telegram', 'qq': 'QQ', 'web': 'Web'}.get(context.platform, '未知')
        platform_ctx = f"\n\n【当前平台】{platform_label}"
        skill_instructions = self._skills_manager.get_skill_instructions()
        system_prompt = self._system_prompt_core + platform_ctx + skill_instructions

        logger.info(f"📋 System prompt length: {len(system_prompt)} chars")
        logger.info(f"OpenAI timeout configured: {self._runtime.timeout}s")

        messages: List[Dict] = [{"role": "system", "content": system_prompt}]
        self._append_history(messages, context.history)

        messages.append({
            "role": "system",
            "content": (
                "━━━ 当前请求边界 ━━━\n"
                "以上为历史记录（用于理解上下文）。\n"
                "以下是用户刚发的新消息，是本轮唯一需要处理的请求。"
            ),
        })
        messages.append({
            "role": "user",
            "content": f"[当前请求] {message}{context.reply_context}",
        })

        tools = self._skills_manager.get_tools() if self._skills_manager.has_tools() else None
        conv_key = (context.platform, context.user_id)
        current_max_tokens = self._initial_max_tokens(message)
        seen_tool_calls: Dict[tuple, int] = {}
        empty_response_retries = 0

        for iteration in range(max_iterations):
            call_kwargs: Dict = {
                "model": self._runtime.model,
                "messages": messages,
                "max_tokens": current_max_tokens,
                "temperature": self._runtime.temperature,
            }
            if tools:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = "auto"

            logger.info(f"Calling {self._runtime.model} (iter {iteration + 1}/{max_iterations})")
            started_at = time.monotonic()
            response = await client.chat.completions.create(**call_kwargs)
            elapsed = time.monotonic() - started_at
            self._log_usage(response)

            if not response.choices:
                return "呜呜，AI 好像没有回复 qwq 要不再试一次？"

            choice = response.choices[0]
            tool_call_count = len(choice.message.tool_calls or [])
            content = choice.message.content or ""
            logger.info(
                f"Model response: finish_reason={choice.finish_reason} "
                f"tool_calls={tool_call_count} content_len={len(content)} elapsed={elapsed:.1f}s"
            )

            if choice.finish_reason == "length":
                if current_max_tokens < self._runtime.max_tokens_cap:
                    current_max_tokens = min(current_max_tokens * 2, self._runtime.max_tokens_cap)
                    logger.warning(f"Response truncated, retrying with max_tokens={current_max_tokens}")
                    messages.append({
                        "role": "user",
                        "content": "[系统] 你上一次的输出因过长被截断，以上查询结果已完整获取。请勿重新查询，直接根据已有数据继续调用下一步工具完成任务。",
                    })
                    continue
                logger.warning("Response truncated even at max cap")
                return "呜呜，回复太长被截断了 qwq 请把任务拆小一点再试试～"

            if not choice.message.tool_calls:
                if content.strip():
                    return content
                if empty_response_retries < 1:
                    empty_response_retries += 1
                    logger.warning("Model returned empty final content, retrying once")
                    messages.append({
                        "role": "user",
                        "content": "[系统] 你上一次没有生成任何可见回复。请不要重新查询，直接根据已有工具结果回复用户；如需继续操作，请调用下一步工具。",
                    })
                    continue
                logger.error("Model returned empty final content twice")
                return "呜呜，AI 返回了空回复 qwq 请再说一次要我怎么处理。"

            parsed_tool_calls = self._parse_tool_calls(choice.message.tool_calls)
            if parsed_tool_calls is None:
                if current_max_tokens < self._runtime.max_tokens_cap:
                    current_max_tokens = min(current_max_tokens * 2, self._runtime.max_tokens_cap)
                    logger.warning(f"Tool args truncated, retrying with max_tokens={current_max_tokens}")
                    messages.append({
                        "role": "user",
                        "content": "[系统] 你上一次生成的工具调用参数因过长被截断。请勿重新查询，直接根据已有数据重新生成完整的工具调用。",
                    })
                    continue
                logger.error("Tool args truncated even at max cap")
                return "呜呜，AI 返回的工具参数格式错误 qwq 请把任务拆小一点再试试～"

            assistant_msg: Dict = {
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc, _ in parsed_tool_calls
                ],
            }
            reasoning_content = getattr(choice.message, 'reasoning_content', None)
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            messages.append(assistant_msg)

            for tc, fn_args in parsed_tool_calls:
                fn_name = tc.function.name
                logger.info(f"Tool call: {fn_name}({fn_args})")
                try:
                    result_str = await self._call_tool_once(
                        fn_name,
                        fn_args,
                        ToolContext(context.platform, context.user_id),
                        seen_tool_calls,
                    )
                except DuplicateToolCallAbort:
                    return "呜呜，AI 陷入了循环 qwq 请换个方式描述任务再试试～"

                try:
                    result_data = json.loads(result_str)
                    if result_data.get("not_bound"):
                        return self._bind_help_text
                    self._save_pending_tool_confirm(conv_key, fn_name, fn_args, result_data)
                except Exception:
                    pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": result_str,
                })

            continue

        return "呜呜，处理太久了 qwq 要不再试一次？"

    def _append_history(self, messages: List[Dict], history: Optional[List[Dict]]) -> None:
        if not history:
            return

        now = datetime.now()
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            ago = ""
            if timestamp:
                try:
                    diff = now - datetime.fromisoformat(timestamp)
                    seconds = int(diff.total_seconds())
                    if seconds < 60:
                        ago = f"{seconds}s ago"
                    elif seconds < 3600:
                        ago = f"{seconds // 60}m ago"
                    elif seconds < 86400:
                        ago = f"{seconds // 3600}h ago"
                    else:
                        ago = f"{seconds // 86400}d ago"
                except Exception:
                    pass
            if role == "user" and ago:
                messages.append({"role": role, "content": f"[{ago}] {content}"})
            else:
                messages.append({"role": role, "content": content})

    def _initial_max_tokens(self, message: str) -> int:
        line_count = message.count("\n") + 1
        return max(self._runtime.max_tokens, min(line_count * 200 + 500, self._runtime.max_tokens_cap))

    def _log_usage(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        cache_hit = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
        cache_miss = getattr(usage, 'prompt_cache_miss_tokens', 0) or 0
        if cache_hit or cache_miss:
            logger.info(f"Cache: hit={cache_hit} miss={cache_miss} tokens")

    def _parse_tool_calls(self, tool_calls: List[object]) -> Optional[List[tuple]]:
        parsed_tool_calls = []
        for tool_call in tool_calls:
            try:
                parsed_tool_calls.append((tool_call, json.loads(tool_call.function.arguments)))
            except json.JSONDecodeError:
                return None
        return parsed_tool_calls

    async def _call_tool_once(
        self,
        fn_name: str,
        fn_args: Dict,
        tool_context: ToolContext,
        seen_tool_calls: Dict[tuple, int],
    ) -> str:
        call_fingerprint = (fn_name, json.dumps(fn_args, sort_keys=True, ensure_ascii=False))
        duplicate_count = seen_tool_calls.get(call_fingerprint, 0)
        if duplicate_count > 0:
            if duplicate_count >= 4:
                logger.error(f"Tool call {fn_name} duplicated {duplicate_count} times, aborting")
                raise DuplicateToolCallAbort()
            logger.warning(f"Duplicate tool call ({duplicate_count}): {fn_name}, injecting forcing hint")
            write_tools = frozenset({
                "keytao_batch_add_to_draft", "keytao_create_phrase",
                "keytao_submit_batch", "keytao_batch_remove_draft_items",
                "keytao_remove_draft_item", "keytao_recall_batch",
            })
            if fn_name in write_tools:
                duplicate_hint = (
                    f"工具 {fn_name} 已执行过，数据已写入。"
                    "禁止重复调用。请直接根据上方执行结果回复用户。"
                )
            else:
                duplicate_hint = (
                    f"工具 {fn_name} 已调用过，结果已在上方消息中。"
                    "禁止再次调用此工具。请直接使用上方已有数据继续下一步操作。"
                )
            seen_tool_calls[call_fingerprint] = duplicate_count + 1
            return json.dumps({"error": "重复调用，已忽略", "message": duplicate_hint}, ensure_ascii=False)

        seen_tool_calls[call_fingerprint] = 1
        return await self._tool_executor.call(fn_name, fn_args, tool_context)

    def _save_pending_tool_confirm(
        self,
        conv_key: tuple,
        fn_name: str,
        fn_args: Dict,
        result_data: Dict,
    ) -> None:
        if fn_name not in ("keytao_create_phrase", "keytao_submit_batch"):
            return
        if not result_data.get("requiresConfirmation"):
            return

        saved = {
            key: value for key, value in fn_args.items()
            if key not in ("confirmed", "platform", "platform_id")
        }
        self._state_store.set(conv_key, PendingToolConfirm(function_name=fn_name, args=saved))
        logger.info(f"💾 Saved PendingToolConfirm: {fn_name}({saved})")
