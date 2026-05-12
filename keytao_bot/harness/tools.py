"""Tool execution adapter for the agent harness."""
import json
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from nonebot.log import logger


@dataclass(frozen=True)
class ToolContext:
    platform: Optional[str] = None
    user_id: Optional[str] = None


class ToolExecutor:
    """Executes registered skills and injects platform context when needed."""

    def __init__(self, get_tool_function: Callable[[str], Optional[Callable]], context_tools: frozenset[str]):
        self._get_tool_function = get_tool_function
        self._context_tools = context_tools

    async def call(self, tool_name: str, arguments: Dict, context: ToolContext) -> str:
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
