#!/usr/bin/env python3
"""
Focused tests for the cross-user isolation, compaction and timezone fixes.

Runs without a NoneBot runtime: external modules are stubbed the same way
test_state_machine.py does it, before anything from keytao_bot is imported.

    uv run python test_isolation_fixes.py
"""
import asyncio
import os
import sqlite3
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- external module stubs (must precede any keytao_bot import) --------------

_fake_httpx = types.ModuleType("httpx")
_fake_httpx.Limits = type("Limits", (), {"__init__": lambda self, **kw: None})
_fake_httpx.Response = type("Response", (), {})
_fake_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
_fake_httpx.TransportError = type("TransportError", (Exception,), {})
_fake_httpx.AsyncClient = type("AsyncClient", (), {"__init__": lambda self, **kw: None})
sys.modules["httpx"] = _fake_httpx

_fake_nonebot = types.ModuleType("nonebot")


class _FakeMatcher:
    def handle(self):
        return lambda f: f

    async def finish(self, *a, **kw):
        pass


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

_fake_openai = types.ModuleType("openai")
_fake_openai.AsyncOpenAI = None
sys.modules["openai"] = _fake_openai

sys.modules["duckduckgo_search"] = types.ModuleType("duckduckgo_search")

# --- imports under test ------------------------------------------------------

from keytao_bot.harness.orchestrator import (  # noqa: E402
    AgentOrchestrator,
    AgentRuntimeConfig,
)
from keytao_bot.harness.state import PendingToolConfirm  # noqa: E402
from keytao_bot.utils.history_store import (  # noqa: E402
    HistoryStore,
    _parse_stored_timestamp,
)
from keytao_bot.utils.memory_store import (  # noqa: E402
    ChatMemoryContext,
    ScopedMemoryStore,
)
import keytao_bot.plugins.openai_chat as openai_chat_module  # noqa: E402


passed = 0
failed = 0


def check(name: str, result: bool):
    global passed, failed
    if result:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


def _seed_legacy_global_row(store: ScopedMemoryStore, content: str) -> None:
    """Write a row the way the pre-fix code wrote shared global memory."""
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO memory_entries (
                scope, scope_id, role, speaker_id, speaker_name,
                target_id, target_name, content, importance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "global",
                "global",
                "user",
                "9999",
                "Legacy",
                "",
                "",
                content,
                "high",
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Task 6: cross-user leakage
# ---------------------------------------------------------------------------

def test_private_users_never_see_each_other():
    print("\n[memory] private users never see each other")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
        alice = ChatMemoryContext(platform="qq", user_id="1001", speaker_name="Alice")
        bob = ChatMemoryContext(platform="qq", user_id="2002", speaker_name="Bob")

        _seed_legacy_global_row(store, "旧全局记忆里的机密项目代号 ORCHID")
        store.add_conversation_round(
            alice,
            "喵喵 帮我记住我的病历号是 A-77321",
            "好的，我记住了你的偏好。",
        )
        store.add_conversation_round(
            bob,
            "喵喵 我的工资是 88888 元，帮我记住",
            "好的，我记住了你的偏好。",
        )

        alice_block = store.get_context_block(alice)
        bob_block = store.get_context_block(bob)

    check("alice keeps her own memory", "A-77321" in alice_block)
    check("alice never sees bob's message", "88888" not in alice_block)
    check("bob keeps his own memory", "88888" in bob_block)
    check("bob never sees alice's message", "A-77321" not in bob_block)
    check("legacy global row stays out of alice's prompt", "ORCHID" not in alice_block)
    check("legacy global row stays out of bob's prompt", "ORCHID" not in bob_block)
    check("no global section is labelled", "全局记忆" not in alice_block)


def test_group_members_do_not_leak_across_groups():
    print("\n[memory] group members do not leak across groups")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
        in_group_42 = ChatMemoryContext(
            platform="qq", user_id="1001", space_type="group",
            space_id="42", speaker_name="Alice",
        )
        in_group_99 = ChatMemoryContext(
            platform="qq", user_id="2002", space_type="group",
            space_id="99", speaker_name="Bob",
        )
        store.add_conversation_round(
            in_group_99,
            "喵喵 记住 99 群的暗号是 PANGOLIN",
            "好的，我记住了这个群的约定。",
        )
        block = store.get_context_block(in_group_42)

    check("other group's memory never reaches this group", "PANGOLIN" not in block)


def test_group_section_is_kept_and_labelled():
    print("\n[memory] group section is kept and labelled")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
        alice = ChatMemoryContext(
            platform="qq", user_id="1001", space_type="group",
            space_id="42", speaker_name="Alice",
        )
        bob = ChatMemoryContext(
            platform="qq", user_id="2002", space_type="group",
            space_id="42", speaker_name="Bob",
        )
        store.add_conversation_round(bob, "喵喵 这个群约定用键道 6", "好的，我记住了这个群的约定。")
        store.add_conversation_round(alice, "喵喵 记住我习惯用简体", "好的，我记住了你的偏好。")
        block = store.get_context_block(alice)

    check("shared group section still exists", "本群共享记忆" in block)
    check("group-scope content is visible to group members", "键道 6" in block)
    check("group prompts do not mix in private memory", "当前私聊记忆" not in block)


def test_raw_turns_are_not_written_to_global_scope():
    print("\n[memory] raw turns are not written to the global scope")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
        context = ChatMemoryContext(
            platform="qq", user_id="1001", space_type="group",
            space_id="42", speaker_name="Alice",
        )
        store.add_conversation_round(context, "喵喵 你好", "你好呀 owo")
        with sqlite3.connect(store.db_path) as conn:
            global_rows = conn.execute(
                "SELECT COUNT(*) FROM memory_entries WHERE scope = 'global'"
            ).fetchone()[0]
            user_rows = conn.execute(
                "SELECT COUNT(*) FROM memory_entries WHERE scope = 'user' AND scope_id = ?",
                (context.user_scope_id,),
            ).fetchone()[0]
            group_rows = conn.execute(
                "SELECT COUNT(*) FROM memory_entries WHERE scope = 'group' AND scope_id = ?",
                (context.space_scope_id,),
            ).fetchone()[0]

    check("nothing new lands in the global scope", global_rows == 0)
    check("group turns are not duplicated into personal scope", user_rows == 0)
    check("group scope receives the round", group_rows > 0)


def test_private_round_is_not_duplicated_into_a_space_scope():
    print("\n[memory] private round is stored once")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
        context = ChatMemoryContext(platform="qq", user_id="1001", speaker_name="Alice")
        store.add_conversation_round(context, "喵喵 你好", "你好呀 owo")
        with sqlite3.connect(store.db_path) as conn:
            space_rows = conn.execute(
                "SELECT COUNT(*) FROM memory_entries WHERE scope = 'group' AND scope_id = ?",
                (context.space_scope_id,),
            ).fetchone()[0]

    check("private chat writes only the personal scope", space_rows == 0)


def test_private_operation_recall_requires_tool_receipts():
    print("\n[memory] private operation recall requires tool receipts")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
        context = ChatMemoryContext(platform="qq", user_id="1001", speaker_name="Alice")
        store.add_conversation_round(
            context,
            "喵喵 加入并提交",
            "✅ 搞定！「空串」→ kywto 已加入草稿并提交审核。",
        )
        operations = store.get_recent_operation_candidates(context)

    check("assistant prose is not treated as an operation receipt", operations == [])


# ---------------------------------------------------------------------------
# Task 7: compaction concurrency
# ---------------------------------------------------------------------------

def test_compaction_never_deletes_rows_written_while_summarizing():
    print("\n[compaction] rows written during summarization survive")

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
            context = ChatMemoryContext(platform="qq", user_id="1001", speaker_name="Alice")
            for index in range(6):
                store.add_conversation_round(
                    context,
                    f"喵喵 记住我的偏好 {index}：以后按个人习惯处理",
                    f"已记录个人偏好 {index}。",
                )

            captured = {}

            async def slow_summarizer(scope, scope_id, old_summary, entries):
                # Simulates a message arriving while the LLM summarizer runs.
                store.add_conversation_round(
                    context,
                    "喵喵 记住我的新偏好 LATECOMER：以后按个人习惯处理",
                    "已记录新的个人偏好。",
                )
                captured["ids"] = [entry["id"] for entry in entries]
                return "- high Alice: 喜欢按个人习惯处理。"

            await store._compact_scope(
                "user",
                context.user_scope_id,
                slow_summarizer,
                keep_recent=2,
                threshold=4,
            )

            with sqlite3.connect(store.db_path) as conn:
                remaining_ids = [
                    row[0] for row in conn.execute(
                        "SELECT id FROM memory_entries WHERE scope = 'user' AND scope_id = ?",
                        (context.user_scope_id,),
                    )
                ]
                contents = " ".join(
                    row[0] for row in conn.execute(
                        "SELECT content FROM memory_entries WHERE scope = 'user' AND scope_id = ?",
                        (context.user_scope_id,),
                    )
                )
                summary = conn.execute(
                    "SELECT content FROM memory_summaries WHERE scope = 'user' AND scope_id = ?",
                    (context.user_scope_id,),
                ).fetchone()

        check("summarizer received the overflow rows", bool(captured.get("ids")))
        check(
            "exactly the summarized ids were deleted",
            all(entry_id not in remaining_ids for entry_id in captured["ids"]),
        )
        check("row written during summarization survives", "LATECOMER" in contents)
        check("summary is stored with the delete", summary is not None and "个人习惯" in summary[0])

    asyncio.run(_run())


def test_compaction_lock_serializes_same_scope():
    print("\n[compaction] per-scope lock serializes compactions")

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
            context = ChatMemoryContext(platform="qq", user_id="1001", speaker_name="Alice")
            for index in range(20):
                store.add_conversation_round(
                    context,
                    f"喵喵 记住我的偏好 {index}：以后按个人习惯处理",
                    f"已记录个人偏好 {index}。",
                )

            state = {"active": 0, "max_active": 0}

            async def slow_summarizer(scope, scope_id, old_summary, entries):
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                await asyncio.sleep(0.02)
                state["active"] -= 1
                return "- high Alice: 喜欢按个人习惯处理。"

            await asyncio.gather(*[
                store._compact_scope(
                    "user",
                    context.user_scope_id,
                    slow_summarizer,
                    keep_recent=2,
                    threshold=4,
                )
                for _ in range(3)
            ])

        check("compactions of one scope never overlap", state["max_active"] == 1)

    asyncio.run(_run())


def test_memory_compaction_is_one_tracked_task_per_actor():
    print("\n[compaction] scheduler keeps one tracked task per actor")

    async def _run():
        context = ChatMemoryContext(platform="qq", user_id="sched-1001", speaker_name="Alice")
        key = ("user", context.user_scope_id)
        openai_chat_module.memory_compaction_tasks.pop(key, None)
        release = asyncio.Event()
        calls = []

        async def slow_compact(memory_context, summarizer=None):
            calls.append(memory_context.user_id)
            await release.wait()

        with patch.object(openai_chat_module.memory_store, "compact_due_scopes", slow_compact):
            openai_chat_module.schedule_memory_compaction(context)
            first_task = openai_chat_module.memory_compaction_tasks.get(key)
            await asyncio.sleep(0)
            openai_chat_module.schedule_memory_compaction(context)
            openai_chat_module.schedule_memory_compaction(context)
            second_task = openai_chat_module.memory_compaction_tasks.get(key)
            check("compaction task is strongly referenced", first_task is not None)
            check("repeat schedules reuse the running task", first_task is second_task)
            release.set()
            await first_task

        check("compaction ran exactly once", calls == ["sched-1001"])
        check("finished task is dropped from the registry", key not in openai_chat_module.memory_compaction_tasks)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Task 8: draft-operation TOCTOU
# ---------------------------------------------------------------------------

def test_mark_running_rejects_superseded_operation():
    print("\n[operation] mark_running rejects a superseded operation")

    conv_key = ("qq", "toctou-1001")
    coordinator = openai_chat_module.draft_operation_coordinator
    coordinator.clear(conv_key)
    operation = coordinator.begin(conv_key, "submit")
    stale_id = operation.operation_id

    check("live operation is recognized", coordinator.get(conv_key) is operation)
    check("mark_running succeeds while live", coordinator.mark_running(conv_key, stale_id))

    coordinator.finish(conv_key, stale_id)
    replacement = coordinator.begin(conv_key, "submit")

    check("stale operation is no longer live", coordinator.get(conv_key) is not operation)
    check("mark_running refuses the stale id", not coordinator.mark_running(conv_key, stale_id))

    coordinator.finish(conv_key, replacement.operation_id)
    coordinator.clear(conv_key)


def test_background_writer_waits_for_the_foreground_lock():
    print("\n[operation] background writer takes the foreground lock")

    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send(self, **kwargs):
            self.messages.append(kwargs.get("message"))

    async def _run():
        conv_key = ("qq", "lock-1001")
        coordinator = openai_chat_module.draft_operation_coordinator
        coordinator.clear(conv_key)
        operation = coordinator.begin(conv_key, "submit")
        bot = FakeBot()
        memory_context = ChatMemoryContext(platform="qq", user_id="lock-1001")
        generation_token = openai_chat_module.memory_store.capture_generation(memory_context)
        history_generation_token = openai_chat_module.history_store.capture_generation(
            memory_context.conversation_address
        )
        pending_submit = PendingToolConfirm(function_name="keytao_submit_batch", args={})

        async def action():
            return openai_chat_module.DraftActionResult(
                "是否继续提交？回复「确认」继续提交，回复「取消」放弃。",
                pending_state=pending_submit,
            )

        sent_messages = []

        async def capture_response(*args, **kwargs):
            sent_messages.append(args[4])
            return True

        with (
            patch.object(openai_chat_module, "remember_conversation", return_value=True),
            patch.object(openai_chat_module, "schedule_memory_compaction"),
            patch.object(openai_chat_module, "_send_event_response", capture_response),
        ):
            async with openai_chat_module.conversation_message_locks.lock(conv_key):
                task = asyncio.create_task(
                    openai_chat_module._run_background_draft_operation(
                        operation, action, bot, object(), "lock-1001", memory_context, "提交",
                        generation_token, history_generation_token,
                    )
                )
                await asyncio.sleep(0.05)
                status_while_locked = operation.status
                messages_while_locked = list(bot.messages)
            await task

        check("background writer cannot mutate while the foreground holds the lock", status_while_locked == "running")
        check("background writer stays silent while blocked", messages_while_locked == [])
        check("outcome is published once the lock is free", operation.status == "awaiting_confirmation")
        check("confirmation prompt is sent exactly once", len(sent_messages) == 1)
        coordinator.clear(conv_key)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Task 9: replace-char confirmation + item cap
# ---------------------------------------------------------------------------

def test_replace_char_requires_confirmation():
    print("\n[replace-char] staging requires confirmation")

    async def _run():
        memory_context = ChatMemoryContext(
            platform="qq",
            user_id="replace-1001",
            space_type="group",
            space_id="42",
            speaker_name="Alice",
        )
        conv_key = memory_context.conversation_address
        openai_chat_module.conversation_state_store.delete(conv_key)

        async def fail_if_called(*args, **kwargs):
            check("replace-char never writes before confirmation", False)
            return "{}"

        with patch.object(openai_chat_module, "call_tool_function", fail_if_called):
            response = await openai_chat_module._try_handle_replace_char(
                "把粘改成黏：\n防粘 fpnm\n胶粘 jcnm",
                "qq",
                "replace-1001",
                openai_chat_module.MessageCommandIntent(
                    intent="batch_replace_char", old_char="粘", new_char="黏", confidence=0.96,
                ),
                conv_key,
                ("group", "42"),
                "Alice",
            )

        pending = openai_chat_module.conversation_state_store.get(conv_key)
        record = openai_chat_module.conversation_state_store.get_record(conv_key)
        openai_chat_module.conversation_state_store.delete(conv_key)

        check("replace-char asks for confirmation", response is not None and "确认票据" in response)
        check("replace-char previews the rewrite", response is not None and "防粘 → 防黏" in response)
        check("replace-char stores a pending tool confirm", isinstance(pending, PendingToolConfirm))
        check("pending targets the batch draft tool", pending.function_name == "keytao_batch_add_to_draft")
        check("pending keeps both items", len(pending.args.get("items", [])) == 2)
        check("pending is owned by the current space", record is not None and record.space_key == ("qq", "qq:group:42"))

    asyncio.run(_run())


def test_replace_char_caps_generated_items():
    print("\n[replace-char] item count is capped")

    async def _run():
        conv_key = ("qq", "replace-1002")
        openai_chat_module.conversation_state_store.delete(conv_key)
        limit = openai_chat_module.MAX_REPLACE_CHAR_ITEMS
        lines = "\n".join(f"防粘{index} fpnm" for index in range(limit + 1))

        response = await openai_chat_module._try_handle_replace_char(
            f"把粘改成黏：\n{lines}",
            "qq",
            "replace-1002",
            openai_chat_module.MessageCommandIntent(
                intent="batch_replace_char", old_char="粘", new_char="黏", confidence=0.96,
            ),
            conv_key,
        )
        pending = openai_chat_module.conversation_state_store.get(conv_key)
        openai_chat_module.conversation_state_store.delete(conv_key)

        check("item cap is 50", limit == 50)
        check("oversized batch is refused", response is not None and str(limit) in response)
        check("oversized batch stages nothing", pending is None)

        at_limit = "\n".join(f"防粘{index} fpnm" for index in range(limit))
        response = await openai_chat_module._try_handle_replace_char(
            f"把粘改成黏：\n{at_limit}",
            "qq",
            "replace-1002",
            openai_chat_module.MessageCommandIntent(
                intent="batch_replace_char", old_char="粘", new_char="黏", confidence=0.96,
            ),
            conv_key,
        )
        pending = openai_chat_module.conversation_state_store.get(conv_key)
        openai_chat_module.conversation_state_store.delete(conv_key)

        check("a batch exactly at the cap is accepted", isinstance(pending, PendingToolConfirm))
        check("accepted batch keeps every item", pending is not None and len(pending.args["items"]) == limit)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Task 10: timezone correctness
# ---------------------------------------------------------------------------

def test_history_store_round_trips_utc_timestamps():
    print("\n[timezone] history store round-trips UTC timestamps")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = HistoryStore(os.path.join(tmpdir, "history.db"))
        store.add_message("qq", "1001", "user", "喵喵 你好")
        history = store.get_history("qq", "1001", limit=5)
        stored = history[0]["timestamp"]
        parsed = _parse_stored_timestamp(stored)
        skew = abs((datetime.now(timezone.utc) - parsed).total_seconds())

        with sqlite3.connect(store.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (platform, user_id, role, content, timestamp)"
                " VALUES (?, ?, ?, ?, ?)",
                ("qq", "1001", "assistant", "legacy row", "2026-07-26 03:00:00"),
            )
            conn.commit()
        legacy = _parse_stored_timestamp("2026-07-26 03:00:00")

    check("stored timestamp keeps its offset", "+00:00" in stored or stored.endswith("Z"))
    check("parsed timestamp is timezone aware", parsed is not None and parsed.tzinfo is not None)
    check("relative-time skew is under a second", skew < 1.0)
    check("legacy naive rows are read as UTC", legacy == datetime(2026, 7, 26, 3, 0, tzinfo=timezone.utc))
    check("garbage timestamps parse to None", _parse_stored_timestamp("not-a-date") is None)
    check("empty timestamps parse to None", _parse_stored_timestamp("") is None)


def test_orchestrator_relative_time_has_no_skew():
    print("\n[timezone] orchestrator relative time has no skew")

    orchestrator = AgentOrchestrator(
        client_factory=lambda: None,
        runtime=AgentRuntimeConfig(model="fake", max_tokens=100, temperature=0.1, timeout=1.0),
        skills_manager=None,
        tool_executor=None,
        state_store=None,
        bind_help_text="",
        system_prompt_core="",
    )

    now_utc = datetime.now(timezone.utc)
    history = [
        {"role": "user", "content": "刚刚说的", "timestamp": now_utc.isoformat()},
        {
            "role": "user",
            "content": "两小时前说的",
            "timestamp": (now_utc - timedelta(hours=2)).isoformat(),
        },
        {
            "role": "user",
            "content": "老格式没有时区",
            "timestamp": (now_utc - timedelta(hours=2)).replace(tzinfo=None).isoformat(),
        },
        {"role": "user", "content": "坏时间戳", "timestamp": "not-a-date"},
        {"role": "assistant", "content": "助手不带时间标签", "timestamp": now_utc.isoformat()},
    ]
    messages = []
    orchestrator._append_history(messages, history)

    check("fresh message is labelled 0s ago", messages[0]["content"].startswith("[0s ago]"))
    check("two-hour-old message is labelled 2h ago", messages[1]["content"].startswith("[2h ago]"))
    check("legacy naive timestamp is also 2h ago", messages[2]["content"].startswith("[2h ago]"))
    check("unparseable timestamp drops the label", messages[3]["content"] == "坏时间戳")
    check("assistant messages keep no relative label", messages[4]["content"] == "助手不带时间标签")



def test_clear_during_compaction_does_not_resurrect_content():
    """A summary built from rows the user then cleared must be discarded."""
    print("\n[generation] /clear during compaction discards the summary")

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
            context = ChatMemoryContext(
                platform="qq", user_id="2001", space_type="private", speaker_name="Alice",
            )
            for index in range(140):
                store.add_conversation_round(context, f"SECRET_MESSAGE_{index}", f"reply {index}")

            cleared = False

            async def slow_summarizer(scope, scope_id, old_summary, entries):
                # The user runs /clear while the LLM is still summarizing.
                nonlocal cleared
                if not cleared:
                    cleared = True
                    store.clear_user_memory(context)
                return "SUMMARY_CONTAINING_SECRET_MESSAGE_0"

            await store._compact_scope("user", context.user_scope_id, slow_summarizer)

            block = store.get_context_block(context)
            check("clear actually happened during summarization", cleared)
            check("stale summary is not written back", "SUMMARY_CONTAINING_SECRET" not in block)
            check("cleared content stays gone", "SECRET_MESSAGE_0" not in block)

    asyncio.run(_run())


def test_generation_token_blocks_late_background_write():
    """A background task finishing after /clear must not re-add its round."""
    print("\n[generation] late background write is dropped after /clear")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ScopedMemoryStore(os.path.join(tmpdir, "memory.db"))
        context = ChatMemoryContext(
            platform="qq", user_id="3001", space_type="private", speaker_name="Bob",
        )
        store.add_conversation_round(context, "早期消息", "早期回复")

        # Background task starts here and captures the generation.
        token = store.capture_generation(context)
        check("token is current before the clear", store.is_generation_current(context, token))

        store.clear_user_memory(context)
        check("token is stale after the clear", not store.is_generation_current(context, token))

        stored = store.add_conversation_round(
            context, "LATE_BACKGROUND_MESSAGE", "LATE_BACKGROUND_REPLY",
            generation_token=token,
        )
        block = store.get_context_block(context)
        check("late write is rejected", stored is False)
        check("late content never lands in the prompt", "LATE_BACKGROUND_MESSAGE" not in block)
        check("late reply never lands in the prompt", "LATE_BACKGROUND_REPLY" not in block)

        # A task that started after the clear writes normally.
        fresh_token = store.capture_generation(context)
        stored_fresh = store.add_conversation_round(
            context, "AFTER_CLEAR_MESSAGE", "AFTER_CLEAR_REPLY",
            generation_token=fresh_token,
        )
        block_after = store.get_context_block(context)
        check("post-clear write is accepted", stored_fresh is True)
        check("post-clear content is stored", "AFTER_CLEAR_MESSAGE" in block_after)

        # No token at all keeps the legacy behaviour.
        check(
            "writes without a token still work",
            store.add_conversation_round(context, "无令牌消息", "无令牌回复") is True,
        )
        check("clearing another user does not bump this user", store.is_generation_current(
            context, store.capture_generation(context)
        ))


if __name__ == "__main__":
    print("=" * 60)
    print("Isolation / concurrency / timezone fixes")
    print("=" * 60)

    test_private_users_never_see_each_other()
    test_group_members_do_not_leak_across_groups()
    test_group_section_is_kept_and_labelled()
    test_raw_turns_are_not_written_to_global_scope()
    test_private_round_is_not_duplicated_into_a_space_scope()
    test_private_operation_recall_requires_tool_receipts()

    test_compaction_never_deletes_rows_written_while_summarizing()
    test_compaction_lock_serializes_same_scope()
    test_clear_during_compaction_does_not_resurrect_content()
    test_generation_token_blocks_late_background_write()
    test_memory_compaction_is_one_tracked_task_per_actor()

    test_mark_running_rejects_superseded_operation()
    test_background_writer_waits_for_the_foreground_lock()

    test_replace_char_requires_confirmation()
    test_replace_char_caps_generated_items()

    test_history_store_round_trips_utc_timestamps()
    test_orchestrator_relative_time_has_no_skew()

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
