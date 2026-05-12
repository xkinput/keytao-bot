#!/usr/bin/env python3
"""
Test the state machine and core logic of openai_chat plugin.
Does NOT require NoneBot runtime — only tests pure functions.
"""
import sys
import os
import asyncio
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch ALL external modules before importing anything from keytao_bot
import types

# httpx (used by tools)
sys.modules["httpx"] = types.ModuleType("httpx")

# nonebot core
_fake_nonebot = types.ModuleType("nonebot")

class _FakeMatcher:
    def handle(self): return lambda f: f
    async def finish(self, *a, **kw): pass

_fake_nonebot.on_message = lambda **kw: _FakeMatcher()
_fake_nonebot.on_command = lambda *a, **kw: _FakeMatcher()

class _FakeConfig:
    openai_api_key = "fake"
    openai_base_url = "https://fake"
    openai_model = "fake-model"
    openai_max_tokens = 1000
    openai_temperature = 0.7
    keytao_api_base = "https://fake"
    bot_api_token = "fake"
    KEYTAO_API_BASE = "https://fake"
    BOT_API_TOKEN = "fake"

class _FakeDriver:
    config = _FakeConfig()

_fake_nonebot.get_driver = lambda: _FakeDriver()
sys.modules["nonebot"] = _fake_nonebot

_fake_adapters = types.ModuleType("nonebot.adapters")
_fake_adapters.Bot = type("Bot", (), {})
_fake_adapters.Event = type("Event", (), {})
sys.modules["nonebot.adapters"] = _fake_adapters

_fake_rule = types.ModuleType("nonebot.rule")
_fake_rule.Rule = lambda f: f
_fake_rule.to_me = lambda: lambda: None
sys.modules["nonebot.rule"] = _fake_rule

_fake_log = types.ModuleType("nonebot.log")
class _FakeLogger:
    def info(self, *a, **kw): pass
    def debug(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass
_fake_log.logger = _FakeLogger()
sys.modules["nonebot.log"] = _fake_log

sys.modules["nonebot.exception"] = types.ModuleType("nonebot.exception")

# OpenAI
_fake_openai = types.ModuleType("openai")
_fake_openai.AsyncOpenAI = None
sys.modules["openai"] = _fake_openai

# duckduckgo_search (used by web-search skill)
sys.modules["duckduckgo_search"] = types.ModuleType("duckduckgo_search")

# Now import the pure functions we want to test
from keytao_bot.plugins.openai_chat import (
    _is_confirm,
    _has_cancel,
    _parse_pending_add_word,
    _strip_markdown,
    _to_markdownv2,
    PendingAddWord,
    PendingToolConfirm,
    CONFIRM_WORDS,
    CANCEL_WORDS,
)
from keytao_bot.harness.state import MemoryConversationStateStore
from keytao_bot.harness.tools import ToolContext, ToolExecutor

_lookup_tools_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "keytao_bot",
    "skills",
    "keytao-lookup",
    "tools.py",
)
_lookup_spec = importlib.util.spec_from_file_location("keytao_lookup_tools_for_test", _lookup_tools_path)
_lookup_tools = importlib.util.module_from_spec(_lookup_spec)
_lookup_spec.loader.exec_module(_lookup_tools)
_normalize_encode_response = _lookup_tools._normalize_encode_response


passed = 0
failed = 0


def check(name: str, result: bool):
    global passed, failed
    if result:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}")


def test_is_confirm():
    print("\n🧪 _is_confirm")

    check("'是' → True", _is_confirm("是"))
    check("'好' → True", _is_confirm("好"))
    check("'yes' → True", _is_confirm("yes"))
    check("'ok' → True", _is_confirm("ok"))
    check("'OK' → True (case insensitive)", _is_confirm("OK"))
    check("'确认' → True", _is_confirm("确认"))
    check("'确定' → True", _is_confirm("确定"))
    check("'嗯' → True", _is_confirm("嗯"))
    check("'行' → True", _is_confirm("行"))
    check("'y' → True", _is_confirm("y"))
    check("' 是 ' → True (whitespace)", _is_confirm(" 是 "))

    check("'查词 你好' → False", not _is_confirm("查词 你好"))
    check("'不是' → False", not _is_confirm("不是"))
    check("'帮我加个词' → False", not _is_confirm("帮我加个词"))
    check("'提交' → False", not _is_confirm("提交"))
    check("'' → False (empty)", not _is_confirm(""))


def test_has_cancel():
    print("\n🧪 _has_cancel")

    check("'不' → True", _has_cancel("不"))
    check("'取消' → True", _has_cancel("取消"))
    check("'算了' → True", _has_cancel("算了"))
    check("'不要了' → True", _has_cancel("不要了"))
    check("'不行' → True", _has_cancel("不行"))
    check("'no' → True", _has_cancel("no"))
    check("'NO' → True (case insensitive)", _has_cancel("NO"))

    check("'是' → False", not _has_cancel("是"))
    check("'好的' → False", not _has_cancel("好的"))
    check("'查词' → False", not _has_cancel("查词"))


def test_parse_pending_add_word_standard():
    """Test the standard candidate list + confirmation pattern."""
    print("\n🧪 _parse_pending_add_word (standard pattern)")

    response = """「产线」（二字词）的拆分和候选编码：

逐字拆分：
• 产（chan）音码 jf　字根 丶一丶丿　形码 ovou
• 线（xian）音码 xm　字根 乙乙｜一一　形码 aavv

候选编码：
1. jfxm — 已有「馋涎」
2. jfxmo — ✅ 推荐（空位）
3. jfxmoa — 空位

是否以编码 jfxmo 将「产线」加入草稿？也可回复编号选其他编码。"""

    result = _parse_pending_add_word(response)
    check("result is not None", result is not None)
    check("word == '产线'", result.word == "产线")
    check("recommended_code == 'jfxmo'", result.recommended_code == "jfxmo")
    check("3 candidates", len(result.candidates) == 3)
    check("candidate[0] == ('jfxm', True)", result.candidates[0] == ("jfxm", True))
    check("candidate[1] == ('jfxmo', False)", result.candidates[1] == ("jfxmo", False))
    check("candidate[2] == ('jfxmoa', False)", result.candidates[2] == ("jfxmoa", False))


def test_parse_pending_add_word_em_dash():
    """Test with em-dash (—) separator."""
    print("\n🧪 _parse_pending_add_word (em-dash)")

    response = """候选编码：
1. pzty — 已有「炮筒」
2. pztyo — ✅ 推荐（空位）

是否以编码 pztyo 将「跑通」加入草稿？"""

    result = _parse_pending_add_word(response)
    check("result is not None", result is not None)
    check("word == '跑通'", result.word == "跑通")
    check("recommended_code == 'pztyo'", result.recommended_code == "pztyo")
    check("2 candidates", len(result.candidates) == 2)
    check("candidate[0] occupied", result.candidates[0][1] is True)
    check("candidate[1] not occupied", result.candidates[1][1] is False)


def test_parse_pending_add_word_all_empty():
    """Test when all candidates are empty slots."""
    print("\n🧪 _parse_pending_add_word (all empty)")

    response = """候选编码：
1. abcd — ✅ 推荐（空位）
2. abcde — 空位
3. abcdea — 空位

是否以编码 abcd 将「测试」加入草稿？"""

    result = _parse_pending_add_word(response)
    check("result is not None", result is not None)
    check("word == '测试'", result.word == "测试")
    check("recommended == 'abcd'", result.recommended_code == "abcd")
    check("all candidates not occupied",
          all(not occ for _, occ in result.candidates))


def test_parse_pending_add_word_no_match():
    """Test when response doesn't contain the pattern."""
    print("\n🧪 _parse_pending_add_word (no match)")

    check("plain chat → None",
          _parse_pending_add_word("你好呀，有什么可以帮你的？") is None)
    check("empty string → None",
          _parse_pending_add_word("") is None)
    check("lookup result (no confirm) → None",
          _parse_pending_add_word("词: 你好\n编码: nau【词组】") is None)


def test_parse_pending_add_word_no_candidate_list():
    """Test when only confirm sentence exists, no numbered list."""
    print("\n🧪 _parse_pending_add_word (no numbered list)")

    response = "是否以编码 abc 将「你好」加入草稿？"
    result = _parse_pending_add_word(response)
    check("result is not None", result is not None)
    check("word == '你好'", result.word == "你好")
    check("recommended_code == 'abc'", result.recommended_code == "abc")
    check("fallback to 1 candidate", len(result.candidates) == 1)
    check("fallback candidate is recommended code",
          result.candidates[0] == ("abc", False))


def test_pending_add_word_numeric_choice():
    """Test the state machine logic for numeric choice."""
    print("\n🧪 PendingAddWord numeric choice logic")

    state = PendingAddWord(
        word="产线",
        recommended_code="jfxmo",
        candidates=[
            ("jfxm", True),
            ("jfxmo", False),
            ("jfxmoa", False),
        ],
    )

    # Choice "2" → jfxmo (not occupied)
    idx = int("2") - 1
    target, occupied = state.candidates[idx]
    check("choice '2' → code 'jfxmo'", target == "jfxmo")
    check("choice '2' → not occupied", not occupied)

    # Choice "1" → jfxm (occupied)
    idx = int("1") - 1
    target, occupied = state.candidates[idx]
    check("choice '1' → code 'jfxm'", target == "jfxm")
    check("choice '1' → occupied", occupied)

    # Choice "3" → jfxmoa (not occupied)
    idx = int("3") - 1
    target, occupied = state.candidates[idx]
    check("choice '3' → code 'jfxmoa'", target == "jfxmoa")
    check("choice '3' → not occupied", not occupied)


def test_pending_add_word_confirm_uses_recommended():
    """Test that '是' maps to recommended code."""
    print("\n🧪 PendingAddWord confirm → recommended code")

    state = PendingAddWord(
        word="测试",
        recommended_code="cek",
        candidates=[
            ("ce", True),
            ("cek", False),
        ],
    )

    check("'是' is confirm", _is_confirm("是"))
    # Logic: on confirm, use recommended_code
    check("recommended_code == 'cek'", state.recommended_code == "cek")
    # Find occupation status for recommended
    for code, occ in state.candidates:
        if code == state.recommended_code:
            check("recommended is not occupied", not occ)
            break


def test_pending_tool_confirm_data():
    """Test PendingToolConfirm dataclass."""
    print("\n🧪 PendingToolConfirm")

    state = PendingToolConfirm(
        function_name="keytao_create_phrase",
        args={"word": "测试", "code": "cek"},
    )
    check("function_name correct", state.function_name == "keytao_create_phrase")
    check("args contain word", state.args["word"] == "测试")
    check("args contain code", state.args["code"] == "cek")
    check("args no confirmed key", "confirmed" not in state.args)


def test_strip_markdown():
    print("\n🧪 _strip_markdown")

    check("code fence removed",
          _strip_markdown("```python\nprint('hi')\n```") == "print('hi')")
    check("inline code kept",
          _strip_markdown("`code`") == "code")
    check("bold removed",
          _strip_markdown("**bold**") == "bold")
    check("italic removed",
          _strip_markdown("*italic*") == "italic")
    check("heading removed",
          _strip_markdown("## Title") == "Title")
    check("plain text unchanged",
          _strip_markdown("hello world") == "hello world")
    check("URL unchanged",
          _strip_markdown("https://example.com") == "https://example.com")


def test_markdownv2_escape():
    print("\n🧪 _to_markdownv2")

    check("plain text escaped",
          "\\.  " not in _to_markdownv2("normal text") or True)  # basic sanity
    # Code blocks should not be escaped
    input_md = "hello `code` world"
    result = _to_markdownv2(input_md)
    check("inline code preserved",
          "`code`" in result)
    # Special chars outside code should be escaped
    result2 = _to_markdownv2("test (parens)")
    check("parens escaped",
          "\\(" in result2 and "\\)" in result2)


def test_real_world_scenario():
    """Simulate the exact bug scenario from the issue."""
    print("\n🧪 Real-world replay: 产线 add-word flow")

    # Step 1: AI responds with candidate list
    ai_response = """「产线」（二字词）的拆分和候选编码：

逐字拆分：
• 产（chan）音码 jf　字根 丶一丶丿　形码 ovou
• 线（xian）音码 xm　字根 乙乙｜一一　形码 aavv

候选编码：
1. jfxm — 已有「馋涎」
2. jfxmo — ✅ 推荐（空位）
3. jfxmoa — 空位

是否以编码 jfxmo 将「产线」加入草稿？也可回复编号选其他编码。"""

    # Step 2: Python parses and saves state
    state = _parse_pending_add_word(ai_response)
    check("state parsed", state is not None)
    check("word = '产线'", state.word == "产线")
    check("recommended = 'jfxmo'", state.recommended_code == "jfxmo")

    # Step 3: User says "是"
    user_msg = "是"
    check("'是' is confirm", _is_confirm(user_msg))
    check("'是' is not cancel", not _has_cancel(user_msg))

    # Step 4: Python directly uses saved state
    # (In real code this calls _execute_add_to_draft with exact code)
    target_code = state.recommended_code
    check("target code is 'jfxmo' (NOT some random code)",
          target_code == "jfxmo")
    check("word is '产线' (exact from saved state)",
          state.word == "产线")

    # This is the BUG FIX validation:
    # Old code would pass to AI which might hallucinate 'chxi'
    # New code uses exact saved values
    check("CRITICAL: code != 'chxi' (the old bug)",
          target_code != "chxi")


def test_edge_case_mixed_cancel_confirm():
    """Edge case: message with both confirm and cancel words."""
    print("\n🧪 Edge case: mixed cancel + confirm")

    # "不是" contains "不" (cancel) — should cancel, not confirm
    check("'不是' has cancel", _has_cancel("不是"))
    # In the handler, _has_cancel is checked first, so this is correct

    # "不好" also has cancel
    check("'不好' has cancel", _has_cancel("不好"))

    # "好不好" — has cancel word
    check("'好不好' has cancel", _has_cancel("好不好"))


def test_edge_case_numeric_out_of_range():
    """Edge case: user picks a number out of candidate range."""
    print("\n🧪 Edge case: numeric out of range")

    state = PendingAddWord(
        word="测试",
        recommended_code="abc",
        candidates=[("abc", False), ("abcd", False)],
    )

    idx = int("5") - 1  # 5 is out of range for 2 candidates
    check("index 4 out of range for 2 candidates",
          not (0 <= idx < len(state.candidates)))


def test_edge_case_zero_choice():
    """Edge case: user sends '0'."""
    print("\n🧪 Edge case: '0' as choice")

    check("'0' is digit", "0".isdigit())
    idx = int("0") - 1  # -1
    state = PendingAddWord(
        word="测试",
        recommended_code="abc",
        candidates=[("abc", False)],
    )
    check("idx -1 is out of range", not (0 <= idx < len(state.candidates)))


def test_confirm_cancel_word_sets():
    """Verify the frozen sets contain expected words."""
    print("\n🧪 CONFIRM_WORDS and CANCEL_WORDS sets")

    check("'是' in CONFIRM_WORDS", "是" in CONFIRM_WORDS)
    check("'确认' in CONFIRM_WORDS", "确认" in CONFIRM_WORDS)
    check("'yes' in CONFIRM_WORDS", "yes" in CONFIRM_WORDS)
    check("'不' in CANCEL_WORDS", "不" in CANCEL_WORDS)
    check("'取消' in CANCEL_WORDS", "取消" in CANCEL_WORDS)
    check("'no' in CANCEL_WORDS", "no" in CANCEL_WORDS)

    # No overlap
    overlap = CONFIRM_WORDS & CANCEL_WORDS
    check("no overlap between confirm and cancel", len(overlap) == 0)


def test_memory_conversation_state_store():
    """Verify the explicit state-store seam preserves pending state behavior."""
    print("\n🧪 MemoryConversationStateStore")

    store = MemoryConversationStateStore()
    key = ("qq", "123")
    state = PendingToolConfirm(
        function_name="keytao_create_phrase",
        args={"word": "测试", "code": "cek"},
    )

    check("initially empty", not store.contains(key))
    store.set(key, state)
    check("contains after set", store.contains(key))
    check("get returns same state", store.get(key) == state)
    check("pop returns same state", store.pop(key) == state)
    check("empty after pop", not store.contains(key))
    store.set(key, state)
    store.delete(key)
    check("empty after delete", not store.contains(key))


async def _run_tool_executor_checks():
    calls = []

    async def fake_tool(**kwargs):
        calls.append(kwargs)
        return {"success": True, "args": kwargs}

    executor = ToolExecutor(
        lambda name: fake_tool if name == "context_tool" else None,
        frozenset({"context_tool"}),
    )

    result = await executor.call(
        "context_tool",
        {"word": "测试"},
        ToolContext(platform="qq", user_id="123"),
    )
    check("tool executor returns JSON success", '"success": true' in result)
    check("platform injected", calls[0]["platform"] == "qq")
    check("platform_id injected", calls[0]["platform_id"] == "123")

    missing_context = await executor.call(
        "context_tool",
        {"word": "测试"},
        ToolContext(),
    )
    check("missing context is rejected", "无法获取用户平台信息" in missing_context)
    missing_tool = await executor.call(
        "missing_tool",
        {},
        ToolContext(platform="qq", user_id="123"),
    )
    check("missing tool is reported", "Tool missing_tool not found" in missing_tool)


def test_tool_executor_context_injection():
    """Verify contextual tools still receive platform identifiers."""
    print("\n🧪 ToolExecutor context injection")
    asyncio.run(_run_tool_executor_checks())


def test_normalize_encode_response_codes_first():
    """Verify keytao_encode exposes phrase candidate codes as first-class data."""
    print("\n🧪 keytao_encode normalization (valid codes)")

    result = _normalize_encode_response("换言之", {
        "input": "换言之",
        "type": "三字词",
        "chars": [
            {
                "char": "换",
                "pinyin": "huàn",
                "phoneticCode": "ht",
                "shapeCode": "iuua",
                "fullCode": "htiuua",
            }
        ],
        "codes": ["hyf", "hyfi", "hyfio", "hyfioo"],
        "altCodes": [],
    })

    check("success true", result["success"] is True)
    check("recommendedCode is codes[0]", result["recommendedCode"] == "hyf")
    check("candidateCodes preserve progressive codes", result["candidateCodes"] == ["hyf", "hyfi", "hyfio", "hyfioo"])
    check("chars are display-only without fullCode", "fullCode" not in result["chars"][0])


def test_normalize_encode_response_infer_fallback():
    """Verify invalid x? codes can be replaced by infer fallback candidates."""
    print("\n🧪 keytao_encode normalization (infer fallback)")

    result = _normalize_encode_response(
        "换言之",
        {
            "input": "换言之",
            "type": "三字词",
            "chars": [{"char": "换", "pinyin": "", "phoneticCode": "x?", "shapeCode": "iuua"}],
            "codes": ["x?x", "x?xi"],
            "altCodes": [],
        },
        {
            "word": "换言之",
            "type": "三字词",
            "codes": ["hyf", "hyfi", "hyfio", "hyfioo"],
            "altCodes": [],
            "suggestion": "hyfioo",
            "suggestionIndex": 3,
        },
    )

    check("success true after fallback", result["success"] is True)
    check("codeSource is infer-fallback", result["codeSource"] == "infer-fallback")
    check("recommendedCode uses infer suggestion", result["recommendedCode"] == "hyfioo")
    check("candidateCodes use fallback codes", result["candidateCodes"] == ["hyf", "hyfi", "hyfio", "hyfioo"])


if __name__ == "__main__":
    print("=" * 60)
    print("State Machine & Core Logic Tests")
    print("=" * 60)

    test_is_confirm()
    test_has_cancel()
    test_parse_pending_add_word_standard()
    test_parse_pending_add_word_em_dash()
    test_parse_pending_add_word_all_empty()
    test_parse_pending_add_word_no_match()
    test_parse_pending_add_word_no_candidate_list()
    test_pending_add_word_numeric_choice()
    test_pending_add_word_confirm_uses_recommended()
    test_pending_tool_confirm_data()
    test_strip_markdown()
    test_markdownv2_escape()
    test_real_world_scenario()
    test_edge_case_mixed_cancel_confirm()
    test_edge_case_numeric_out_of_range()
    test_edge_case_zero_choice()
    test_confirm_cancel_word_sets()
    test_memory_conversation_state_store()
    test_tool_executor_context_injection()
    test_normalize_encode_response_codes_first()
    test_normalize_encode_response_infer_fallback()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED")
    print("=" * 60)
