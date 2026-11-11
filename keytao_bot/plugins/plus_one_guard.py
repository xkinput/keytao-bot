"""Keep bot-directed messages and commands out of the plus-one matcher."""

from nonebot.adapters import Bot, Event
from nonebot.exception import IgnoredException
from nonebot.matcher import Matcher
from nonebot.message import run_preprocessor
from nonebot.typing import T_State


_group_streaks = {}


@run_preprocessor
async def guard_plus_one(bot: Bot, event: Event, matcher: Matcher, state: T_State) -> None:
    if matcher.plugin_name != "nonebot_plugin_plus_one":
        return
    message = getattr(event, "original_message", None) or event.get_message()
    reply_sender = getattr(getattr(event, "reply", None), "sender", None)
    # TrieRule resolves commands before the priority-1 plus-one matcher runs.
    if (
        event.is_tome()
        or state.get("_prefix", {}).get("command")
        or any(segment.type == "at" and str(segment.data.get("qq")) == bot.self_id
               for segment in message)
        or str(getattr(reply_sender, "user_id", "")) == bot.self_id
    ):
        raise IgnoredException("Bot-directed message or command")
    if hasattr(event, "group_id"):
        from nonebot_plugin_plus_one.handler import is_equal

        group_id = str(event.group_id)
        message = event.get_message()
        last_message, count = _group_streaks.get(group_id, (None, 0))
        count = count + 1 if last_message is not None and is_equal(last_message, message) else 1
        _group_streaks[group_id] = (message, count)
        if count > 2:
            raise IgnoredException("Already repeated this streak")
