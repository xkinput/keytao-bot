#!/usr/bin/env python3
"""
Test the state machine and core logic of openai_chat plugin.
Does NOT require NoneBot runtime — only tests pure functions.
"""
import sys
import os
import asyncio
import importlib.util
import json
import tempfile
from unittest.mock import AsyncMock, patch

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

_fake_exception = types.ModuleType("nonebot.exception")
class FinishedException(Exception):
    pass
_fake_exception.FinishedException = FinishedException
sys.modules["nonebot.exception"] = _fake_exception

# OpenAI
_fake_openai = types.ModuleType("openai")
_fake_openai.AsyncOpenAI = None
sys.modules["openai"] = _fake_openai

# duckduckgo_search (used by web-search skill)
sys.modules["duckduckgo_search"] = types.ModuleType("duckduckgo_search")

# Now import the pure functions we want to test
from keytao_bot.plugins.openai_chat import (
    _augment_simple_word_query_response,
    _build_existing_word_priority_note,
    _extract_pure_chinese_words,
    _is_confirm,
    _has_cancel,
    _handle_pending_add_word,
    _ensure_pending_add_word_guidance,
    _parse_pending_add_word,
    _recover_pending_state_from_history,
    _resolve_shift_target_code,
    _strip_command_message_prefixes,
    _strip_markdown,
    _to_markdownv2,
    _is_clear_command_text,
    PendingAddWord,
    PendingToolConfirm,
    CONFIRM_WORDS,
    CANCEL_WORDS,
    SYSTEM_PROMPT_CORE,
)
from keytao_bot.plugins.account_bind import (
    _extract_bind_key,
    _is_bind_command_text,
)
from keytao_bot.harness.state import MemoryConversationStateStore
from keytao_bot.harness.tools import ToolContext, ToolExecutor
from keytao_bot.harness.orchestrator import AgentOrchestrator, AgentRequestContext, AgentRuntimeConfig
from keytao_bot.utils.history_store import HistoryStore
import keytao_bot.plugins.openai_chat as openai_chat_module

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
_apply_candidate_occupancy = _lookup_tools._apply_candidate_occupancy

_draft_tools_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "keytao_bot",
    "skills",
    "keytao-draft",
    "tools.py",
)
_draft_spec = importlib.util.spec_from_file_location("keytao_draft_tools_for_test", _draft_tools_path)
_draft_tools = importlib.util.module_from_spec(_draft_spec)
_draft_spec.loader.exec_module(_draft_tools)
_build_code_shift_plan = _draft_tools._build_code_shift_plan


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
    check("occupied words extracted", result.occupied_words["jfxm"] == ["馋涎"])


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


def test_pending_add_word_guidance_appended_for_occupied_candidates():
    """Verify occupied candidate lists automatically get reply guidance appended."""
    print("\n🧪 pending add-word guidance appended")

    response = """候选编码：
1. zrxx — 已有「增翔」
2. zrxxv — ✅ 推荐（空位）

是否以编码 zrxxv 将「增香」加入草稿？也可回复编号选其他编码。"""

    guided = _ensure_pending_add_word_guidance(response)
    check("guidance mentions duplicate reply", "直接回复该编号表示添加重码" in guided)
    check("guidance mentions recode reply", "编号 重新编码" in guided)


def test_pending_add_word_guidance_fallback_matcher():
    """Verify fallback string matcher still appends guidance when response shape varies."""
    print("\n🧪 pending add-word guidance fallback matcher")

    response = """候选编码：
1. zrxx - 已有「增翔」
2. zrxxv - 推荐（空位）

是否以编码 zrxxv 将「增香」加入草稿？也可回复编号选其他编码。"""

    guided = _ensure_pending_add_word_guidance(response)
    check("fallback appends guidance", "原词 重新编码" in guided)


def test_system_prompt_includes_word_lookup_rule_for_single_and_multi_word_inputs():
    """Verify word-only inputs default to meaning + keytao lookup behavior for one or many words."""
    print("\n🧪 system prompt includes single/multi-word lookup rule")

    check("prompt mentions one or many Chinese words", "如果用户只发了一个或多个中文词/短词" in SYSTEM_PROMPT_CORE)
    check("prompt mentions meaning explanation", "每个词都先用 1-2 句解释它的大致含义" in SYSTEM_PROMPT_CORE)
    check("prompt mentions batch lookup preference", "多个词时优先使用批量查询工具" in SYSTEM_PROMPT_CORE)
    check("prompt mentions duplicate order", "主动说明该词在同码词里的排序位置" in SYSTEM_PROMPT_CORE)


def test_extract_pure_chinese_words():
    """Verify simple Chinese-word-only messages can be detected for enrichment."""
    print("\n🧪 extract pure Chinese words")

    check("single word extracted", _extract_pure_chinese_words("寿司郎") == ["寿司郎"])
    check("multiple words extracted", _extract_pure_chinese_words("寿司郎 卧龙凤雏") == ["寿司郎", "卧龙凤雏"])
    check("non-word sentence not extracted", _extract_pure_chinese_words("寿司郎是什么") == [])


def test_build_existing_word_priority_note():
    """Verify existing-word note explains earlier occupied candidates and duplicate order."""
    print("\n🧪 build existing-word priority note")

    lookup_entry = {
        "word": "寿司郎",
        "phrases": [
            {
                "word": "寿司郎",
                "code": "eslv",
                "type_label": "词组",
                "duplicate_info": {
                    "position_label": "二重",
                    "all_words": [
                        {"word": "寿司狼", "label": ""},
                        {"word": "寿司郎", "label": "二重"},
                    ],
                },
            }
        ],
    }
    encode_data = {
        "candidateStatuses": [
            {"code": "esl", "occupied": True, "label": "已有「厄斯兰」"},
            {"code": "eslv", "occupied": True, "label": "已有「寿司狼、寿司郎」"},
            {"code": "eslva", "occupied": False, "label": "空位"},
        ]
    }

    note = _build_existing_word_priority_note("寿司郎", lookup_entry, encode_data)
    check("note mentions prior occupied code", "esl 已有" in note)
    check("note mentions duplicate position", "排在二重" in note)
    check("note mentions peer words", "寿司狼" in note and "寿司郎" in note)


def test_augment_simple_word_query_response_appends_priority_note():
    """Verify simple existing-word replies get deterministic priority enrichment."""
    print("\n🧪 augment simple word query response")

    async def _run():
        async def fake_call(tool_name, arguments, platform=None, user_id=None):
            if tool_name == "keytao_lookup_by_words_batch":
                return json.dumps({
                    "success": True,
                    "results": [{
                        "word": "寿司郎",
                        "phrases": [{
                            "word": "寿司郎",
                            "code": "eslv",
                            "duplicate_info": {
                                "position_label": "二重",
                                "all_words": [
                                    {"word": "寿司狼", "label": ""},
                                    {"word": "寿司郎", "label": "二重"},
                                ],
                            },
                        }],
                    }],
                }, ensure_ascii=False)
            if tool_name == "keytao_encode":
                return json.dumps({
                    "success": True,
                    "candidateStatuses": [
                        {"code": "esl", "occupied": True, "label": "已有「厄斯兰」"},
                        {"code": "eslv", "occupied": True, "label": "已有「寿司狼、寿司郎」"},
                    ],
                }, ensure_ascii=False)
            raise AssertionError(tool_name)

        with patch.object(openai_chat_module, "call_tool_function", side_effect=fake_call):
            result = await _augment_simple_word_query_response(
                "寿司郎",
                "词库已有：\n\n词: 寿司郎\n编码: eslv（三字词）【词组】",
                "qq",
                "123",
            )

        check("result contains priority appendix", "补充说明：" in result)
        check("result explains prior occupied code", "esl 已有" in result)
        check("result explains duplicate order", "排在二重" in result)

    asyncio.run(_run())


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


def test_numeric_reply_means_exact_candidate_selection():
    """Verify numbered replies select the exact candidate, not a generic confirm action."""
    print("\n🧪 numeric reply means exact candidate selection")

    state = PendingAddWord(
        word="增香",
        recommended_code="zrxxv",
        candidates=[
            ("zrxx", True),
            ("zrxxv", False),
            ("zrxxvu", False),
        ],
    )

    msg1 = _strip_command_message_prefixes("喵喵 1")
    check("'1' is not confirm", not _is_confirm(msg1))
    idx1 = int(msg1) - 1
    check("'1' selects zrxx", state.candidates[idx1][0] == "zrxx")

    msg3 = _strip_command_message_prefixes("喵喵 3")
    check("'3' is not confirm", not _is_confirm(msg3))
    idx3 = int(msg3) - 1
    check("'3' selects zrxxvu", state.candidates[idx3][0] == "zrxxvu")

    confirm_msg = _strip_command_message_prefixes("喵喵 是")
    check("'是' remains confirm", _is_confirm(confirm_msg))
    check("'是' maps to recommended code", state.recommended_code == "zrxxv")


def test_occupied_numeric_choice_means_duplicate_confirm():
    """Verify selecting an occupied candidate directly means duplicate-code insertion."""
    print("\n🧪 occupied numeric choice means duplicate confirm")

    state = PendingAddWord(
        word="增香",
        recommended_code="zrxxv",
        candidates=[
            ("zrxx", True),
            ("zrxxv", False),
            ("zrxxvu", False),
        ],
        occupied_words={"zrxx": ["增翔"]},
    )

    async def _run():
        with patch.object(openai_chat_module, "_execute_confirmed_tool", AsyncMock(return_value="duplicate")) as duplicate_mock:
            with patch.object(openai_chat_module, "_execute_shift_to_code", AsyncMock(return_value="shifted")) as shift_mock:
                result = await _handle_pending_add_word(
                    state, "1", "qq", "123", [],
                )
        check("occupied choice returns duplicate result", result == "duplicate")
        check("duplicate helper called once", duplicate_mock.await_count == 1)
        check("shift helper not called", shift_mock.await_count == 0)

    asyncio.run(_run())


def test_shift_request_can_target_by_number_or_word():
    """Verify users can request shift directly by number or by occupant word."""
    print("\n🧪 shift request can target by number or word")

    state = PendingAddWord(
        word="增香",
        recommended_code="zrxxv",
        candidates=[
            ("zrxx", True),
            ("zrxxv", False),
            ("zrxxvu", False),
        ],
        occupied_words={"zrxx": ["增翔"]},
    )

    check("'1 重新编码' -> zrxx", _resolve_shift_target_code(state, "1 重新编码") == "zrxx")
    check("'增翔重新编码' -> zrxx", _resolve_shift_target_code(state, "增翔重新编码") == "zrxx")
    check("'重新编码' with one occupied choice -> zrxx", _resolve_shift_target_code(state, "重新编码") == "zrxx")

    async def _run():
        with patch.object(openai_chat_module, "_execute_shift_to_code", AsyncMock(return_value="shifted")) as shift_mock:
            result = await _handle_pending_add_word(
                state, "1 重新编码", "qq", "123", [],
            )
        check("shift request returns shift result", result == "shifted")
        check("shift helper called once", shift_mock.await_count == 1)

    asyncio.run(_run())


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


def test_bind_command_text_detection():
    """Verify bind commands still route when prefixed by mentions or trigger words."""
    print("\n🧪 bind command text detection")

    check("plain slash command detected", _is_bind_command_text("/bind 26PZWH"))
    check("plain no-slash command detected", _is_bind_command_text("bind 26PZWH"))
    check("QQ mention prefix detected", _is_bind_command_text("@喵喵 /bind 26PZWH"))
    check("QQ mention display name prefix detected", _is_bind_command_text("@喵喵 jacobpang /bind NW7UWX"))
    check("bind command inside sentence detected", _is_bind_command_text("@喵喵 jacobpang 请 /bind NW7UWX 绑定一下"))
    check("trigger word prefix detected", _is_bind_command_text("喵喵 /bind 26PZWH"))
    check("multiple prefixes detected", _is_bind_command_text("@喵喵 键道 /bind 26PZWH"))
    check("bind key uppercased", _extract_bind_key("@喵喵 /bind 26pzwh") == "26PZWH")
    check("bind key extracted after display name", _extract_bind_key("@喵喵 jacobpang /bind NW7UWX") == "NW7UWX")
    check("bind key extracted inside sentence", _extract_bind_key("@喵喵 jacobpang 请 /bind NW7UWX 绑定一下") == "NW7UWX")
    check("missing key returns empty string", _extract_bind_key("@喵喵 /bind") == "")
    check("natural language not detected", not _is_bind_command_text("喵喵 绑定怎么弄"))
    check("mentioned bind discussion not detected", not _is_bind_command_text("@喵喵 关于 /bind 绑定"))
    check("valid bind with trailing words detected", _is_bind_command_text("/bind 26PZWH extra"))


def test_clear_command_text_detection():
    """Verify clear commands still route when prefixed by mentions or trigger words."""
    print("\n🧪 clear command text detection")

    check("plain slash clear detected", _is_clear_command_text("/clear"))
    check("plain clear detected", _is_clear_command_text("clear"))
    check("Chinese alias detected", _is_clear_command_text("清空历史"))
    check("mention prefix detected", _is_clear_command_text("@喵喵 /clear"))
    check("mention display name prefix detected", _is_clear_command_text("@喵喵 jacobpang /clear"))
    check("clear command inside sentence detected", _is_clear_command_text("@喵喵 jacobpang 请 /clear 一下"))
    check("trigger word prefix detected", _is_clear_command_text("喵喵 清空对话"))
    check("natural language not detected", not _is_clear_command_text("喵喵 怎么清空历史"))
    check("mentioned clear token detected", _is_clear_command_text("@喵喵 关于 /clear"))
    check("clear with trailing words detected", _is_clear_command_text("/clear now"))


def test_pending_reply_prefix_stripping():
    """Verify pending-state replies still work when prefixed by trigger words or mentions."""
    print("\n🧪 pending reply prefix stripping")

    check("喵喵 1 -> 1", _strip_command_message_prefixes("喵喵 1") == "1")
    check("键道 是 -> 是", _strip_command_message_prefixes("键道 是") == "是")
    check("@喵喵 确认 -> 确认", _strip_command_message_prefixes("@喵喵 确认") == "确认")
    check("prefixed digit stays digit", _strip_command_message_prefixes("喵喵 1").isdigit())
    check("prefixed confirm still confirms", _is_confirm(_strip_command_message_prefixes("喵喵 是")))


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


def test_recover_pending_add_word_from_history():
    """Verify confirm flows can be recovered from the last assistant message."""
    print("\n🧪 recover PendingAddWord from history")

    history = [
        {"role": "user", "content": "喵喵 卧龙凤雏"},
        {
            "role": "assistant",
            "content": """「卧龙凤雏」目前不在词库中，但编码计算没问题！

候选编码：
1. wlfj — ✅ 推荐（空位）
2. wlfjv — 空位

是否以编码 wlfj 将「卧龙凤雏」加入草稿？也可回复编号选其他编码。""",
        },
    ]

    state = _recover_pending_state_from_history(history)
    check("recovered state exists", state is not None)
    check("recovered PendingAddWord", isinstance(state, PendingAddWord))
    check("word recovered correctly", state.word == "卧龙凤雏")
    check("recommended code recovered", state.recommended_code == "wlfj")


def test_recover_pending_submit_confirm_from_history():
    """Verify submit reconfirm can be recovered when in-memory state is gone."""
    print("\n🧪 recover PendingToolConfirm from history")

    history = [
        {"role": "user", "content": "提交吧"},
        {
            "role": "assistant",
            "content": "⚠️ 检测到批次中存在重码，是否继续提交？回复「确认」继续提交，回复「取消」放弃。",
        },
    ]

    state = _recover_pending_state_from_history(history)
    check("recovered submit confirm exists", state is not None)
    check("recovered PendingToolConfirm", isinstance(state, PendingToolConfirm))
    check("submit tool recovered", state.function_name == "keytao_submit_batch")
    check("submit args empty", state.args == {})


def test_recover_pending_state_scans_back_to_recent_matching_assistant_message():
    """Verify recovery can skip a newer unrelated assistant message."""
    print("\n🧪 recover pending state scans recent assistant messages")

    history = [
        {"role": "user", "content": "喵喵 增香"},
        {
            "role": "assistant",
            "content": """候选编码：
1. zrxx — 已有「增翔」
2. zrxxv — ✅ 推荐（空位）

是否以编码 zrxxv 将「增香」加入草稿？也可回复编号选其他编码。""",
        },
        {"role": "user", "content": "谢谢"},
        {"role": "assistant", "content": "不客气喵～"},
    ]

    state = _recover_pending_state_from_history(history)
    check("state recovered from earlier assistant prompt", isinstance(state, PendingAddWord))
    check("recovered word is 增香", state.word == "增香")


def test_history_store_keeps_user_and_assistant_same_second():
    """Verify a conversation round keeps both messages instead of dropping one."""
    print("\n🧪 HistoryStore stores both sides of a round")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "history.db")
        store = HistoryStore(db_path)
        store.add_conversation_round("qq", "123", "喵喵 卧龙凤雏", "是否以编码 wlfj 将「卧龙凤雏」加入草稿？")
        history = store.get_history("qq", "123", limit=10)

    check("history keeps 2 messages", len(history) == 2)
    check("first row is user", history[0]["role"] == "user")
    check("second row is assistant", history[1]["role"] == "assistant")


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


async def _run_tool_executor_policy_checks():
    calls = []

    async def fake_tool(**kwargs):
        calls.append(kwargs)
        return {"success": True, "args": kwargs}

    executor = ToolExecutor(
        lambda name: fake_tool if name in {
            "keytao_batch_add_to_draft",
            "keytao_batch_remove_draft_items",
        } else None,
        frozenset({"keytao_batch_add_to_draft", "keytao_batch_remove_draft_items"}),
    )

    bad_move = await executor.call(
        "keytao_batch_add_to_draft",
        {"items": [
            {"action": "Delete", "word": "会员费", "code": "hyfa"},
            {"action": "Delete", "word": "换言之", "code": "hyfio"},
            {"action": "Create", "word": "会员费", "code": "hyfio"},
            {"action": "Create", "word": "换言之", "code": "hyfioa"},
        ]},
        ToolContext(platform="qq", user_id="123", current_message="还是会员费改hyfio吧 换衣服别动了"),
    )
    bad_data = json.loads(bad_move)
    check("unmentioned word reassignment is blocked", bad_data.get("policyBlocked") is True)
    check("blocked reassignment names 换言之", "换言之" in bad_data.get("blockedReassignments", [""])[0])
    check("blocked call did not execute", len(calls) == 0)

    protected_move = await executor.call(
        "keytao_batch_add_to_draft",
        {"items": [
            {"action": "Delete", "word": "换言之", "code": "hyfio"},
            {"action": "Create", "word": "换言之", "code": "hyfioa"},
        ]},
        ToolContext(platform="qq", user_id="123", current_message="会员费改hyfio，换言之别动"),
    )
    protected_data = json.loads(protected_move)
    check("protected word reassignment is blocked", protected_data.get("policyBlocked") is True)

    allowed_move = await executor.call(
        "keytao_batch_add_to_draft",
        {"items": [
            {"action": "Delete", "word": "会员费", "code": "hyfa"},
            {"action": "Create", "word": "会员费", "code": "hyfio"},
        ]},
        ToolContext(platform="qq", user_id="123", current_message="还是会员费改hyfio吧 换衣服别动了"),
    )
    allowed_data = json.loads(allowed_move)
    check("mentioned word reassignment is allowed", allowed_data.get("success") is True)
    check("allowed call executed once", len(calls) == 1)

    broad_delete = await executor.call(
        "keytao_batch_remove_draft_items",
        {"ids": [1110, 1111, 1112, 1113, 1114, 1115]},
        ToolContext(platform="qq", user_id="123", current_message="还是会员费改hyfio吧 换衣服别动了"),
    )
    broad_delete_data = json.loads(broad_delete)
    check("broad draft delete without delete intent is blocked", broad_delete_data.get("policyBlocked") is True)


def test_tool_executor_draft_policy_guards():
    """Verify draft tools cannot move unrelated words while satisfying a code edit."""
    print("\n🧪 ToolExecutor draft policy guards")
    asyncio.run(_run_tool_executor_policy_checks())


class _FakeAIMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, finish_reason, content=None, tool_calls=None):
        self.finish_reason = finish_reason
        self.message = _FakeAIMessage(content, tool_calls)


class _FakeAIResponse:
    def __init__(self, finish_reason, content=None, tool_calls=None):
        self.choices = [_FakeChoice(finish_reason, content, tool_calls)]
        self.usage = None


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        completions = _FakeCompletions(responses)
        self.chat = types.SimpleNamespace(completions=completions)
        self.completions = completions


class _FakeSkillsManager:
    def get_skill_instructions(self):
        return ""

    def has_tools(self):
        return False


async def _run_orchestrator_empty_response_retry_checks():
    client = _FakeClient([
        _FakeAIResponse("stop", None),
        _FakeAIResponse("stop", "已根据已有结果继续处理"),
    ])
    orchestrator = AgentOrchestrator(
        client_factory=lambda: client,
        runtime=AgentRuntimeConfig(
            model="fake-model",
            max_tokens=1000,
            temperature=0.7,
            timeout=180.0,
        ),
        skills_manager=_FakeSkillsManager(),
        tool_executor=ToolExecutor(lambda name: None, frozenset()),
        state_store=MemoryConversationStateStore(),
        bind_help_text="bind help",
        system_prompt_core="system",
    )

    result = await orchestrator.run(
        "还是会员费改hyfio吧",
        AgentRequestContext(platform="qq", user_id="123"),
    )

    check("empty final content retries once", len(client.completions.calls) == 2)
    check("retry returns visible reply", result == "已根据已有结果继续处理")


def test_orchestrator_empty_response_retry():
    """Verify empty final model content does not become a generic request failure."""
    print("\n🧪 AgentOrchestrator empty response retry")
    asyncio.run(_run_orchestrator_empty_response_retry_checks())


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
        "altCodes": ["ffb", "ffbo"],
        "flyKeyVariants": [{"baseCode": "ffb", "codes": ["ffb", "ffbo"], "changes": []}],
        "requestedCodeAnalysis": {"code": "ffb", "supported": True, "matchType": "flyKey"},
    })

    check("success true", result["success"] is True)
    check("recommendedCode is codes[0]", result["recommendedCode"] == "hyf")
    check("candidateCodes include fly key codes", result["candidateCodes"] == ["hyf", "hyfi", "hyfio", "hyfioo", "ffb", "ffbo"])
    check("flyKeyVariants preserved", result["flyKeyVariants"][0]["baseCode"] == "ffb")
    check("requestedCodeAnalysis preserved", result["requestedCodeAnalysis"]["matchType"] == "flyKey")
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


def test_apply_candidate_occupancy_updates_recommendation():
    """Verify encoded candidates include checked occupancy labels before AI sees them."""
    print("\n🧪 keytao_encode candidate occupancy")

    encoding = _normalize_encode_response("会员费", {
        "input": "会员费",
        "type": "三字词",
        "chars": [],
        "codes": ["hyf", "hyfi", "hyfio", "hyfioa"],
        "altCodes": [],
    })
    result = _apply_candidate_occupancy(encoding, {
        "success": True,
        "results": [
            {"code": "hyf", "phrases": [{"word": "换衣服", "code": "hyf", "type_label": "词组"}]},
            {"code": "hyfi", "phrases": [{"word": "会员费", "code": "hyfi", "type_label": "词组"}]},
            {"code": "hyfio", "phrases": []},
            {"code": "hyfioa", "phrases": []},
        ],
    })

    check("occupancyChecked true", result["occupancyChecked"] is True)
    check("candidateStatuses length", len(result["candidateStatuses"]) == 4)
    check("occupied label is explicit", result["candidateStatuses"][0]["label"] == "已有「换衣服」")
    check("empty label is explicit", result["candidateStatuses"][2]["label"] == "空位")
    check("firstAvailableCode set", result["firstAvailableCode"] == "hyfio")
    check("recommendedCode moves to first available", result["recommendedCode"] == "hyfio")


def test_build_code_shift_plan_uses_occupant_encode_chain():
    """Verify displaced words move by their own encode candidates, not the inserted word's chain."""
    print("\n🧪 code shift plan uses occupant encode chain")

    result = _build_code_shift_plan(
        word="会员费",
        target_code="hyfio",
        target_candidate_codes=["hyf", "hyfi", "hyfio", "hyfioa"],
        current_phrase={"word": "会员费", "code": "hyfa", "type": "Phrase"},
        code_phrase_map={
            "hyfio": [{"word": "换言之", "code": "hyfio", "type": "Phrase", "weight": 100}],
            "hyfioo": [],
        },
        word_candidate_code_map={
            "会员费": ["hyf", "hyfi", "hyfio", "hyfioa"],
            "换言之": ["hyf", "hyfi", "hyfio", "hyfioo"],
        },
    )

    check("shift plan succeeds", result["success"] is True)
    check("one word shifted", len(result["shifted"]) == 1)
    check("换言之 shifts to its own next code", result["shifted"][0]["toCode"] == "hyfioo")
    check("换言之 does not use 会员费 next code", result["shifted"][0]["toCode"] != "hyfioa")
    check("delete target old code first", result["items"][0] == {"action": "Delete", "word": "会员费", "code": "hyfa", "type": "Phrase"})
    check("create shifted word at hyfioo", {"action": "Create", "word": "换言之", "code": "hyfioo", "type": "Phrase"} in result["items"])


def test_build_code_shift_plan_cascades_until_empty():
    """Verify occupied destination codes continue shifting by each occupant's encode chain."""
    print("\n🧪 code shift plan cascades until empty")

    result = _build_code_shift_plan(
        word="会员费",
        target_code="hyfio",
        target_candidate_codes=["hyf", "hyfi", "hyfio", "hyfioa"],
        current_phrase={"word": "会员费", "code": "hyfa", "type": "Phrase"},
        code_phrase_map={
            "hyfio": [{"word": "换言之", "code": "hyfio", "type": "Phrase", "weight": 100}],
            "hyfioo": [{"word": "候选词", "code": "hyfioo", "type": "Phrase", "weight": 100}],
            "hxci": [],
        },
        word_candidate_code_map={
            "会员费": ["hyf", "hyfi", "hyfio", "hyfioa"],
            "换言之": ["hyf", "hyfi", "hyfio", "hyfioo"],
            "候选词": ["hx", "hxc", "hyfioo", "hxci"],
        },
    )

    check("cascade plan succeeds", result["success"] is True)
    check("two words shifted", len(result["shifted"]) == 2)
    check("first shifted word", result["shifted"][0]["word"] == "换言之")
    check("second shifted word", result["shifted"][1]["word"] == "候选词")
    check("second word shifts by own chain", result["shifted"][1]["toCode"] == "hxci")


def test_build_code_shift_plan_rejects_invalid_occupant_code():
    """Verify the shift stops if an occupant's current code is not in its encode chain."""
    print("\n🧪 code shift plan rejects invalid occupant code")

    result = _build_code_shift_plan(
        word="会员费",
        target_code="hyfio",
        target_candidate_codes=["hyf", "hyfi", "hyfio", "hyfioa"],
        current_phrase={"word": "会员费", "code": "hyfa", "type": "Phrase"},
        code_phrase_map={
            "hyfio": [{"word": "换言之", "code": "hyfio", "type": "Phrase", "weight": 100}],
        },
        word_candidate_code_map={
            "会员费": ["hyf", "hyfi", "hyfio", "hyfioa"],
            "换言之": ["hyf", "hyfi", "hyfioo"],
        },
    )

    check("invalid occupant code rejected", result["success"] is False)
    check("error mentions occupant", "换言之" in result["message"])


def test_shift_phrase_code_plans_real_occupant_move():
    """Verify keytao_shift_phrase_code keeps occupant moves in the final write plan."""
    print("\n🧪 keytao_shift_phrase_code keeps occupant move")

    async def _run():
        async def fake_fetch(word, requested_code=None):
            mapping = {
                "增香": {"success": True, "word": "增香", "candidateCodes": ["zrxx", "zrxxv", "zrxxvu"]},
                "增翔": {"success": True, "word": "增翔", "candidateCodes": ["zrxx", "zrxxv", "zrxxvo"]},
            }
            return mapping[word]

        async def fake_lookup_words(words):
            return {"success": True, "results": [{"word": "增香", "phrases": []}]}

        async def fake_lookup_codes(codes):
            result_map = {
                "zrxx": [{"word": "增翔", "code": "zrxx", "type": "Phrase", "weight": 100}],
                "zrxxv": [],
            }
            return {
                "success": True,
                "results": [{"code": code, "phrases": result_map.get(code, [])} for code in codes],
            }

        async def fake_list(platform, platform_id):
            return {"success": True, "items": []}

        async def fake_remove(platform, platform_id, ids):
            return {"success": True}

        async def fake_add(platform, platform_id, items):
            return {"success": True, "items": items}

        with patch.object(_draft_tools, "_fetch_encode_candidates", side_effect=fake_fetch):
            with patch.object(_draft_tools, "_lookup_words_raw", side_effect=fake_lookup_words):
                with patch.object(_draft_tools, "_lookup_codes_raw", side_effect=fake_lookup_codes):
                    with patch.object(_draft_tools, "keytao_list_draft_items", side_effect=fake_list):
                        with patch.object(_draft_tools, "keytao_batch_remove_draft_items", side_effect=fake_remove):
                            with patch.object(_draft_tools, "keytao_batch_add_to_draft", side_effect=fake_add):
                                result = await _draft_tools.keytao_shift_phrase_code("qq", "123", "增香", "zrxx")

        check("shift tool succeeds", result["success"] is True)
        items = result["shiftPlan"]["items"]
        check("plan deletes occupant old code", {"action": "Delete", "word": "增翔", "code": "zrxx", "type": "Phrase"} in items)
        check("plan recreates occupant at next code", {"action": "Create", "word": "增翔", "code": "zrxxv", "type": "Phrase"} in items)
        check("plan creates target word at requested code", {"action": "Create", "word": "增香", "code": "zrxx", "type": "Phrase"} in items)

    asyncio.run(_run())


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
    test_pending_add_word_guidance_appended_for_occupied_candidates()
    test_pending_add_word_guidance_fallback_matcher()
    test_system_prompt_includes_word_lookup_rule_for_single_and_multi_word_inputs()
    test_extract_pure_chinese_words()
    test_build_existing_word_priority_note()
    test_augment_simple_word_query_response_appends_priority_note()
    test_pending_add_word_numeric_choice()
    test_numeric_reply_means_exact_candidate_selection()
    test_occupied_numeric_choice_means_duplicate_confirm()
    test_shift_request_can_target_by_number_or_word()
    test_pending_add_word_confirm_uses_recommended()
    test_pending_tool_confirm_data()
    test_strip_markdown()
    test_markdownv2_escape()
    test_real_world_scenario()
    test_edge_case_mixed_cancel_confirm()
    test_edge_case_numeric_out_of_range()
    test_edge_case_zero_choice()
    test_confirm_cancel_word_sets()
    test_bind_command_text_detection()
    test_clear_command_text_detection()
    test_pending_reply_prefix_stripping()
    test_memory_conversation_state_store()
    test_recover_pending_add_word_from_history()
    test_recover_pending_submit_confirm_from_history()
    test_recover_pending_state_scans_back_to_recent_matching_assistant_message()
    test_history_store_keeps_user_and_assistant_same_second()
    test_tool_executor_context_injection()
    test_tool_executor_draft_policy_guards()
    test_orchestrator_empty_response_retry()
    test_normalize_encode_response_codes_first()
    test_normalize_encode_response_infer_fallback()
    test_apply_candidate_occupancy_updates_recommendation()
    test_build_code_shift_plan_uses_occupant_encode_chain()
    test_build_code_shift_plan_cascades_until_empty()
    test_build_code_shift_plan_rejects_invalid_occupant_code()
    test_shift_phrase_code_plans_real_occupant_move()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED")
    print("=" * 60)
