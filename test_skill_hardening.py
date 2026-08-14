#!/usr/bin/env python3
"""
Focused regression tests for the skill/plugin hardening work.

Covers per-user API key isolation, URL path-segment validation, the skills
loader (duplicate tool names + partial skill loads), the shared group broadcast
helper, and the broadened per-type draft code validation.

Runs without a NoneBot runtime: every external module is stubbed first, mirroring
the harness at the top of test_state_machine.py.
"""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Stub every external module before importing anything from keytao_bot
# ---------------------------------------------------------------------------

sys.modules["httpx"] = types.ModuleType("httpx")

_fake_nonebot = types.ModuleType("nonebot")


class _FakeMatcher:
    def handle(self):
        return lambda f: f

    async def finish(self, *a, **kw):
        pass


_fake_nonebot.on_message = lambda **kw: _FakeMatcher()
_fake_nonebot.on_command = lambda *a, **kw: _FakeMatcher()


class _FakeConfig:
    keytao_api_base = "https://fake"
    bot_api_token = "fake"
    KEYTAO_API_BASE = "https://fake"
    BOT_API_TOKEN = "fake"


class _FakeDriver:
    config = _FakeConfig()

    def on_shutdown(self, func):
        return func


_fake_nonebot.get_driver = lambda: _FakeDriver()
_fake_nonebot.get_bots = lambda: {}
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
    def info(self, *a, **kw):
        pass

    def debug(self, *a, **kw):
        pass

    def warning(self, *a, **kw):
        pass

    def error(self, *a, **kw):
        pass


_fake_log.logger = _FakeLogger()
sys.modules["nonebot.log"] = _fake_log

_fake_exception = types.ModuleType("nonebot.exception")


class FinishedException(Exception):
    pass


_fake_exception.FinishedException = FinishedException
sys.modules["nonebot.exception"] = _fake_exception

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from keytao_bot.skills import SkillsManager  # noqa: E402
from keytao_bot.utils import group_notify, http_client  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_draft_tools = _load_module(
    "keytao_draft_tools_for_hardening_test",
    _REPO_ROOT / "keytao_bot" / "skills" / "keytao-draft" / "tools.py",
)


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


# ---------------------------------------------------------------------------
# 5. per-user API key isolation
# ---------------------------------------------------------------------------


def test_user_api_key_isolation():
    print("\n🧪 per-user API key isolation")

    old_user_keys = getattr(_FakeConfig, "keytao_user_api_keys", None)
    old_api_key = getattr(_FakeConfig, "keytao_api_key", None)
    old_bot_user_key = getattr(_FakeConfig, "bot_user_api_key", None)
    try:
        _FakeConfig.keytao_user_api_keys = json.dumps({
            "qq:1001": "kt_user_1001",
            "qq:default": "kt_platform_default",
            "default": "kt_global_default",
            "2002": "kt_bare_platform_id",
        })
        # A global admin key must never leak onto another user's request.
        _FakeConfig.keytao_api_key = "kt_admin_global"
        _FakeConfig.bot_user_api_key = "kt_admin_bot"

        check(
            "bound account gets its own key",
            _draft_tools.get_user_api_key("qq", "1001") == "kt_user_1001",
        )
        check(
            "unbound account gets no key at all",
            _draft_tools.get_user_api_key("qq", "9999") is None,
        )
        check(
            "platform default is not a fallback",
            _draft_tools.get_user_api_key("qq", "3003") is None,
        )
        check(
            "bare platform id is not a fallback",
            _draft_tools.get_user_api_key("qq", "2002") is None,
        )
        check(
            "global admin key is not a fallback",
            _draft_tools.get_user_api_key("telegram", "7777") is None,
        )

        bound_headers = _draft_tools.get_bot_headers("qq", "1001")
        unbound_headers = _draft_tools.get_bot_headers("qq", "9999")
        check("bound account sends X-API-Key", bound_headers.get("X-API-Key") == "kt_user_1001")
        check("unbound account sends no X-API-Key", "X-API-Key" not in unbound_headers)
        check("unbound account still sends bot token", unbound_headers.get("X-Bot-Token") == "fake")
    finally:
        _FakeConfig.keytao_user_api_keys = old_user_keys
        _FakeConfig.keytao_api_key = old_api_key
        _FakeConfig.bot_user_api_key = old_bot_user_key


# ---------------------------------------------------------------------------
# 3. path injection
# ---------------------------------------------------------------------------


def test_path_segment_validation():
    print("\n🧪 URL path segment validation")

    check("traversal segment rejected", _draft_tools._safe_path_segment("1/../../admin") is None)
    check("query marker rejected", _draft_tools._safe_path_segment("abc?x=1") is None)
    check("fragment marker rejected", _draft_tools._safe_path_segment("abc#frag") is None)
    check("whitespace rejected", _draft_tools._safe_path_segment("ab cd") is None)
    check("percent escape rejected", _draft_tools._safe_path_segment("%2e%2e") is None)
    check("empty rejected", _draft_tools._safe_path_segment("") is None)
    check("plain id accepted", _draft_tools._safe_path_segment(" batch-123 ") == "batch-123")

    check("numeric id accepts int", _draft_tools._safe_numeric_id(42) == 42)
    check("numeric id rejects traversal", _draft_tools._safe_numeric_id("1/../../admin") is None)
    check("numeric id rejects letters", _draft_tools._safe_numeric_id("abc") is None)
    check("numeric id rejects zero", _draft_tools._safe_numeric_id(0) is None)
    check("numeric id rejects negative", _draft_tools._safe_numeric_id(-5) is None)

    check(
        "unsafe batch id never lands in a URL",
        "admin" not in _draft_tools.make_batch_url("1/../../admin"),
    )


def test_remove_draft_item_rejects_unsafe_pr_id():
    print("\n🧪 keytao_remove_draft_item rejects unsafe ids")

    async def _explode():
        raise AssertionError("no HTTP call may be made for an invalid pr_id")

    original = http_client.get_keytao_client
    http_client.get_keytao_client = _explode
    try:
        traversal = asyncio.run(
            _draft_tools.keytao_remove_draft_item("qq", "1", "1/../../admin")
        )
        letters = asyncio.run(_draft_tools.keytao_remove_draft_item("qq", "1", "abc"))
        batch = asyncio.run(
            _draft_tools.keytao_batch_remove_draft_items("qq", "1", [1, "2/../admin"])
        )
    finally:
        http_client.get_keytao_client = original

    check("traversal pr_id rejected", traversal.get("success") is False)
    check("traversal pr_id reports the bad value", "1/../../admin" in traversal.get("message", ""))
    check("non-numeric pr_id rejected", letters.get("success") is False)
    check("batch delete rejects any invalid id", batch.get("success") is False)


# ---------------------------------------------------------------------------
# 27. skills loader: duplicate names + partial loads
# ---------------------------------------------------------------------------


_TOOLS_TEMPLATE = '''
TOOLS = [
    {{"type": "function", "function": {{"name": "{tool_name}", "description": "d"}}}}
]


async def _impl(**kwargs):
    return {{"skill": "{skill_name}"}}


TOOL_FUNCTIONS = {{"{tool_name}": _impl}}
'''

_BROKEN_TOOLS = "raise RuntimeError('boom: this skill cannot be imported')\n"


def _write_skill(root: Path, skill_name: str, tools_source: str) -> Path:
    skill_dir = root / skill_name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name}\n---\n\ndoc-for-{skill_name}\n", encoding="utf-8"
    )
    (skill_dir / "tools.py").write_text(tools_source, encoding="utf-8")
    return skill_dir


def test_skills_loader_hardening():
    print("\n🧪 skills loader hardening")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = _write_skill(
            root, "skill_first",
            _TOOLS_TEMPLATE.format(tool_name="shared_tool", skill_name="skill_first"),
        )
        second = _write_skill(
            root, "skill_second",
            _TOOLS_TEMPLATE.format(tool_name="shared_tool", skill_name="skill_second"),
        )
        broken = _write_skill(root, "skill_broken", _BROKEN_TOOLS)
        healthy = _write_skill(
            root, "skill_healthy",
            _TOOLS_TEMPLATE.format(tool_name="unique_tool", skill_name="skill_healthy"),
        )

        manager = SkillsManager(skills_dir=str(root))
        # Explicit ordering: directory iteration order is not guaranteed.
        manager.load_skill(first)
        manager.load_skill(second)
        manager.load_skill(broken)
        manager.load_skill(healthy)

        tool_names = [
            tool.get("function", {}).get("name") for tool in manager.get_tools()
        ]
        check("duplicate tool schema registered only once", tool_names.count("shared_tool") == 1)
        check("unique tool still registered", "unique_tool" in tool_names)
        check(
            "first skill keeps ownership of the duplicate name",
            manager.tool_owners.get("shared_tool") == "skill_first",
        )

        winner = asyncio.run(manager.get_tool_function("shared_tool")())
        check("duplicate function does not overwrite the first one", winner == {"skill": "skill_first"})

        check("broken skill registers no tools", manager.tool_owners.get("broken_tool") is None)
        check("broken skill injects no SKILL.md", "skill_broken" not in manager.skill_docs)
        check("healthy skill injects its SKILL.md", "skill_healthy" in manager.skill_docs)
        check("first skill injects its SKILL.md", "skill_first" in manager.skill_docs)
        check(
            "broken skill doc never reaches the system prompt",
            "doc-for-skill_broken" not in manager.get_skill_instructions(),
        )


def test_skills_loader_order_is_deterministic():
    print("\n🧪 skills loader order is deterministic")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_skill(
            root,
            "skill_zeta",
            _TOOLS_TEMPLATE.format(tool_name="zeta_tool", skill_name="skill_zeta"),
        )
        _write_skill(
            root,
            "skill_alpha",
            _TOOLS_TEMPLATE.format(tool_name="alpha_tool", skill_name="skill_alpha"),
        )

        manager = SkillsManager(skills_dir=str(root))
        discovered = [root / "skill_zeta", root / "skill_alpha"]
        with patch.object(Path, "iterdir", return_value=iter(discovered)):
            manager.load_all_skills()

        check(
            "skill block follows name-sorted directory order",
            list(manager.skill_docs) == ["skill_alpha", "skill_zeta"],
        )
        check(
            "tool registration follows the same deterministic skill order",
            [tool["function"]["name"] for tool in manager.get_tools()]
            == ["alpha_tool", "zeta_tool"],
        )


# ---------------------------------------------------------------------------
# 29. shared group broadcast helper
# ---------------------------------------------------------------------------


class _FailingBot:
    def __init__(self, failures: int = 99):
        self.failures = failures
        self.calls = []

    async def send_group_msg(self, group_id, message):
        self.calls.append((group_id, message))
        if len(self.calls) <= self.failures:
            raise RuntimeError("simulated send failure")
        return {"message_id": len(self.calls)}


def test_group_notification_helper():
    print("\n🧪 group broadcast helper")

    long_text = "\n".join(f"line-{index:04d} " + "x" * 60 for index in range(60))
    check("test fixture is longer than one message", len(long_text) > 1500)
    chunks = group_notify.split_message(long_text)
    check("long message is chunked", len(chunks) > 1)
    check("every chunk fits the limit", all(len(chunk) <= group_notify.MAX_MESSAGE_CHARS for chunk in chunks))
    check("chunking loses no content", "".join(chunks).replace("\n", "") == long_text.replace("\n", ""))

    single_line = "y" * 4000
    hard_split = group_notify.split_message(single_line)
    check("oversized single line is hard split", len(hard_split) == 3)
    check("hard split preserves content", "".join(hard_split) == single_line)

    original_delay = group_notify.RETRY_DELAY_SECONDS
    group_notify.RETRY_DELAY_SECONDS = 0.0
    try:
        always_failing = _FailingBot()
        failure = asyncio.run(
            group_notify.send_group_notification("hi", ["123"], bot=always_failing)
        )
        check("permanent failure is reported", failure["failed"] == ["123"])
        check("permanent failure sends nothing", failure["sent"] == [])
        check("send is retried exactly once", len(always_failing.calls) == 2)

        flaky = _FailingBot(failures=1)
        recovered = asyncio.run(
            group_notify.send_group_notification("hi", ["123"], bot=flaky)
        )
        check("retry can recover", recovered["sent"] == ["123"])
        check("recovered send made two attempts", len(flaky.calls) == 2)

        multi = _FailingBot(failures=0)
        broadcast = asyncio.run(
            group_notify.send_group_notification("hi", ["1", "2"], bot=multi)
        )
        check("multiple groups are supported", broadcast["sent"] == ["1", "2"])
    finally:
        group_notify.RETRY_DELAY_SECONDS = original_delay


# ---------------------------------------------------------------------------
# 25. per-type code validation is no longer bypassed by type
# ---------------------------------------------------------------------------


def test_item_code_validation_scope():
    print("\n🧪 per-type draft code validation")

    should = _draft_tools._should_validate_item_code
    check("CSS create is validated", should({"action": "Create", "word": "喵", "code": "mao", "type": "CSS"}))
    check("CSSSingle create is validated", should({"action": "Create", "word": "喵", "code": "m", "type": "CSSSingle"}))
    check("English create is validated", should({"action": "Create", "word": "hello", "code": "hello", "type": "English"}))
    check("Symbol create is validated", should({"action": "Create", "word": "，", "code": ";a", "type": "Symbol"}))
    check("Supplement create is validated", should({"action": "Create", "word": "补", "code": "bu", "type": "Supplement"}))
    check("Link create is validated", should({"action": "Create", "word": "https://a.test", "code": "at", "type": "Link"}))
    check("Change is validated", should({"action": "Change", "word": "新词", "code": "xc", "type": "Phrase"}))
    check("CSS change is validated", should({"action": "Change", "word": "黏", "code": "nm", "type": "CSS"}))
    check("Delete is still skipped", not should({"action": "Delete", "word": "旧词", "code": "jc", "type": "Phrase"}))
    check("incomplete item is skipped", not should({"action": "Create", "word": "", "code": "abc"}))

    async def _run():
        css_ok = await _draft_tools._validate_draft_item_code(
            {"action": "Create", "word": "喵喵", "code": "mm", "type": "CSS"}
        )
        css_long = await _draft_tools._validate_draft_item_code(
            {"action": "Create", "word": "喵喵", "code": "abcdefghijk", "type": "CSS"}
        )
        css_bad_charset = await _draft_tools._validate_draft_item_code(
            {"action": "Create", "word": "喵喵", "code": "ab3", "type": "CSS"}
        )
        english = await _draft_tools._validate_draft_item_code(
            {"action": "Create", "word": "hello", "code": "hello", "type": "English"}
        )
        deleted = await _draft_tools._validate_draft_item_code(
            {"action": "Delete", "word": "旧词", "code": "jc", "type": "Phrase"}
        )
        return css_ok, css_long, css_bad_charset, english, deleted

    css_ok, css_long, css_bad_charset, english, deleted = asyncio.run(_run())

    check("valid CSS code is accepted", css_ok.get("success") is True)
    check("unverifiable CSS needs manual review", css_ok.get("needsManualReview") is True)
    check("over-long CSS code is rejected", css_long.get("success") is False)
    check("non-letter CSS code is rejected", css_bad_charset.get("success") is False)
    check("English item is accepted", english.get("success") is True)
    check("English item needs manual review", english.get("needsManualReview") is True)
    check("Delete item is skipped", deleted.get("skipped") is True)

    # The structured verdict must ride on the item, not only inside the remark.
    stamped = _draft_tools._stamp_item_review_flag(
        {"word": "喵喵", "code": "mm", "type": "CSS"},
        css_ok,
    )
    check("manual review flag is stamped structurally", stamped.get("needsManualReview") is True)
    check("manual review reason is recorded", bool(stamped.get("manualReviewReason")))

    from_remark = _draft_tools._normalize_draft_item_for_request({
        "word": "追速",
        "code": "fbsjuv",
        "type": "Phrase",
        "remark": "喵喵审词：读音 zhui su；自动审核：该词需管理员审核（常用词信号不足）",
    })
    check("legacy remark marker upgrades to a structured flag", from_remark.get("needsManualReview") is True)

    never_downgraded = _draft_tools._stamp_item_review_flag({
        "word": "追速",
        "code": "fbsjuv",
        "needsManualReview": False,
        "remark": "自动审核：该词需管理员审核（常用词信号不足）",
    })
    check("a True signal is never downgraded", never_downgraded.get("needsManualReview") is True)


def main():
    test_user_api_key_isolation()
    test_path_segment_validation()
    test_remove_draft_item_rejects_unsafe_pr_id()
    test_skills_loader_hardening()
    test_skills_loader_order_is_deterministic()
    test_group_notification_helper()
    test_item_code_validation_scope()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    if failed:
        print("❌ FAILURES DETECTED")
    else:
        print("✅ ALL TESTS PASSED")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
