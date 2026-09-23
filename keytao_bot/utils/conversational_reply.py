"""Keep non-operation replies conversational without another model call."""

import re


class ConversationalReply(str):
    """A checked reply; repeated delivery checks must be idempotent."""


_NARRATION = re.compile(
    r"(?:用户|对方|发送者)(?:向我|跟我|给我|在|发(?:来|送)?的|这条|的(?:这条|当前)?(?:消息|意图|请求)|想|希望|需要|要求|只是|是在|正在|说|问)"
    r"|(?:这条|该条|本条|当前)(?:消息|请求).{0,25}(?:问候|打招呼|闲聊|意图|操作)"
    r"|(?:没有|未|无需|不需要|不涉及|不包含|并未).{0,20}(?:命令|操作|执行|写入|工具|补充信息)"
    r"|(?:不|无需|决定不|不会|不必)(?:再|进行|调用|执行|处理).{0,15}(?:工具|操作|请求|任务)"
    r"|(?:等|有了).{0,15}(?:具体需求|具体请求)|(?:路由|分类|意图)(?:判断|结果|决定)"
    r"|(?:只是|属于|是一次).{0,10}(?:问候|寒暄|闲聊)"
)
_WORK_COPY = re.compile(
    r"命令|指令|候选|草稿|证据|可发送|可执行|本次未|本轮未|tool_calls|suggestedCommand"
    r"|(?:可以|你可以|可|请)(?:直接)?(?:发送|回复|输入)|(?:我可以|需要我帮你)"
    r"|(?:^|\n)\s*(?:[-•]|\d+[.、])"
)


def natural_reply(message: str, previous: str = "") -> str:
    """Select wording only AFTER routing has established no operation intent."""
    choices = ("喵，我在听呢。", "嗯嗯，本喵陪你聊～", "喵，听着呢～")
    if re.search(r"晚安|睡了|睡觉", message):
        choices = ("晚安喵，做个好梦～", "好好休息，晚安呀～", "喵，祝你好梦！")
    elif re.search(r"早|上午", message):
        choices = ("早呀，喵～", "早上好呀，今天也精神满满喵！", "喵，早呀！")
    elif re.search(r"在吗|在不在", message):
        choices = ("在呢，喵～", "喵，本喵在这儿！", "在呀，耳朵竖起来啦～")
    elif re.search(r"谢谢|多谢|辛苦", message):
        choices = ("不客气喵～", "嘿嘿，能帮上忙就好～", "喵，收到你的心意啦～")
    elif re.search(r"哈哈|嘿嘿|笑", message):
        choices = ("哈哈，本喵也乐了～", "嘿嘿，喵～", "喵哈哈～")
    return next(choice for choice in choices if choice != previous)


def enforce_conversational_reply(message: str, reply: str, history=()) -> ConversationalReply:
    """Reject the whole meta reply; never leave a routing summary behind."""
    if isinstance(reply, ConversationalReply):
        return reply
    previous = next((str(item.get("content") or "") for item in reversed(history)
                     if item.get("role") == "assistant"), "")
    text = str(reply or "").strip()
    if not text or _NARRATION.search(text) or _WORK_COPY.search(text) or text == previous:
        text = natural_reply(message, previous)
    else:
        text = " ".join(text.splitlines())
    return ConversationalReply(text)
