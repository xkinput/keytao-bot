"""Offline checks proving the E2E production guards fail before dispatch."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch
from urllib.parse import unquote

import httpx

from .recording import ArtifactRecorder, _redact_sensitive
from .scenarios import (
    SCENARIOS,
    S19_ADVERTISED_WORDS,
    S20_BATCH_WORDS,
    S21_BATCH_WORDS,
    S22_BATCH_WORDS,
    S23_BATCH_WORDS,
    S24_NATURAL_ASSENT,
    S24_RECOMMENDED_CODE,
    S24_WORD,
    S25_COMBINED_COMMAND,
    S25_NATURAL_ADD,
    S25_PREFIX_CODE,
    S25_SELECTED_CODE,
    S25_WORD,
    S26_CODE,
    S26_COMMAND,
    S26_OCCUPANT,
    S26_WORD,
    S37_COMMAND,
    S37_OCCUPANT,
    S37_SELECTION,
    S37_TARGET_CODE,
    S37_WORD,
    S38_EXPLICIT_READING_MESSAGE,
    S38_EXPLANATION_MESSAGE,
    S38_NEGATIVE_MODIFIER_MESSAGE,
    S38_QUERY_CONTROLS,
    S39_COMMAND,
    S39_OCCUPANT,
    S39_SELECTION,
    S39_TARGET_CODE,
    S39_WORD,
    S40_BATCH_WORDS,
    S40_COPY_WORD,
    S40_OCCUPANT,
    S40_TARGET_CODE,
    S41_CODE_MESSAGE,
    S41_EXISTING_CODES,
    S41_READING_MESSAGE,
    S41_WORD,
    S42_WORDS,
    S43_WORD,
    S45_CHARACTER_ANSWER,
    S45_CHARACTER_QUESTION,
    S45_FIRST_CODE,
    S45_FIRST_WORD,
    S45_SECOND_CODE,
    S45_SECOND_WORD,
    S45_SWAP_MESSAGE,
    S46_FIRST_CODE,
    S46_FIRST_SHIFTED_CODE,
    S46_MESSAGE,
    S46_OCCUPANT,
    S46_PLAN_COMMAND,
    S46_SECOND_CODE,
    S46_SECOND_SHIFTED_CODE,
    S46_WORD,
    S48_EXPLICIT_EVICTION,
    S48_NUMBERED_EVICTION,
    S48_NUMBERED_RECODE,
    S48_OCCUPANT,
    S48_RECOMMENDED_CODE,
    S48_SHIFTED_CODE,
    S48_TARGET_CODE,
    S48_WORD,
    S50_AFTER,
    S50_COMPOSITE_FRONT,
    S50_CONTEXT,
    S50_CORRECTION,
    S50_DESTINATION,
    S50_DISCOVERY,
    S50_FRONT,
    S50_INITIAL_WORD,
    S50_INITIAL_FREE_CODE,
    S50_OCCUPANT,
    S50_REPLACEMENT_WORD,
    S27_ASSENT,
    S27_META_QUESTION,
    S27_WORD,
    S28_INVALID_CODE,
    S28_WORD,
    S29_CODE,
    S29_CURRENT,
    S32_CHAIN_COMMAND,
    S32_WORD_LIST_COMMAND,
    S33_DISCOVERY,
    S33_EXTERNAL_EXPECTED,
    S33_EXTERNAL_OCCUPANT,
    S33_EXTERNAL_QUERY,
    S33_EXTERNAL_WORDS,
    S33_SIX_WORDS,
    S33_WORDS,
    _s33_external_query_pairs,
    S34_PENDING_CODE,
    S34_WORD,
    S35_FRONT_CASES,
    S35_FREE_CONTROL,
    _recommended_empty_code,
    assert_batch_link_hosts,
    same_unique_item_set,
)
from .run import (
    S9_ZDIC_WARMUP_BACKOFF_SECONDS,
    _encoded_matches_zdic_fixture,
    abort_record_for_error,
    build_bot_reference_fixture,
    collect_local_socket_stats,
    ensure_s9_fixture,
    ensure_s16_fixture,
    ensure_s18_fixture,
    ensure_s25_fixture,
    ensure_s29_fixture,
    ensure_s41_fixture,
    ensure_scenario_zdic_fixture,
    repair_scenario_dictionary_fixture,
)
from .runtime import E2EBotHarness, LocalNextClient, NextServer, RigInfrastructureError
from .safety import (
    BLOCKED_EXTERNAL_DOMAINS,
    EncodeDelayController,
    NetworkAllowlist,
    PronunciationPoisonController,
    SafetyViolation,
    validate_admin_identity,
    validate_keytao_base,
    validate_llm_base,
    validate_next_database_url,
    validate_test_binding,
)
from .zdic_seed import (
    ZDIC_FIXTURES_BY_SCENARIO,
    seed_s9_zdic_cache,
    seed_zdic_cache,
    zdic_cache_rows_for_scenarios,
)


class NextServerRuntimeTests(unittest.TestCase):
    def test_local_http_clients_ignore_system_proxy_settings(self) -> None:
        local_client = LocalNextClient(
            base_url="http://localhost:3100",
            bot_token="test",
        )
        pooled = local_client._pooled_client()
        try:
            self.assertFalse(pooled._trust_env)
        finally:
            asyncio.run(local_client.close())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = NextServer(
                next_dir=root,
                base_url="http://localhost:3100",
                artifact_dir=root,
                start_timeout=1,
                child_env={},
            )
            response = MagicMock(status_code=200)
            response.json.return_value = {
                "phrases": [],
                "pagination": {},
            }
            session = MagicMock()
            session.get = AsyncMock(return_value=response)
            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=session)
            context.__aexit__ = AsyncMock(return_value=None)
            with patch("e2e.runtime.httpx.AsyncClient", return_value=context) as ctor:
                self.assertTrue(asyncio.run(server._probe()))
            ctor.assert_called_once_with(
                timeout=2.0,
                follow_redirects=False,
                trust_env=False,
            )

    def test_runtime_dir_keeps_read_only_source_and_owns_next_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            artifact = root / "artifact"
            runtime_root = root / "runtime"
            (source / ".next" / "dev").mkdir(parents=True)
            (source / "app").mkdir()
            (source / "node_modules").mkdir()
            (source / ".next" / "dev" / "lock").write_text("stale")
            (source / "app" / "page.tsx").write_text("export default null")
            (source / ".env").write_text("SECRET=not-copied")

            server = NextServer(
                next_dir=source,
                base_url="http://localhost:3100",
                artifact_dir=artifact,
                start_timeout=1,
                child_env={},
                runtime_root=runtime_root,
            )
            runtime = server._prepare_runtime_dir()

            self.assertEqual(runtime, runtime_root.resolve() / "keytao-next")
            self.assertFalse(str(runtime).startswith(str(artifact)))
            self.assertFalse((runtime / "app").is_symlink())
            self.assertTrue((runtime / "app" / "page.tsx").is_symlink())
            self.assertEqual(
                (runtime / "app" / "page.tsx").read_text(),
                "export default null",
            )
            self.assertTrue((runtime / "node_modules").is_symlink())
            self.assertTrue((runtime / ".env").is_symlink())
            self.assertFalse((runtime / ".next").exists())
            (runtime / ".next" / "dev").mkdir(parents=True)
            (runtime / ".next" / "dev" / "lock").write_text("live")
            self.assertEqual(
                (source / ".next" / "dev" / "lock").read_text(),
                "stale",
            )

            self.assertEqual(server._prepare_runtime_dir(), runtime)
            self.assertTrue((runtime / ".next" / "dev" / "lock").is_file())

            (source / "app" / "page.tsx").write_text("export default 42")
            refreshed = server._prepare_runtime_dir()
            self.assertEqual(refreshed, runtime)
            self.assertEqual(
                (refreshed / "app" / "page.tsx").read_text(),
                "export default 42",
            )
            self.assertFalse((refreshed / ".next").exists())


class ArtifactRetentionTests(unittest.TestCase):
    def test_retention_keeps_current_and_newest_completed_runs(self) -> None:
        from .run import prune_artifact_runs

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            run_names = [
                f"20260824T0{index}0000Z-{index:08x}"
                for index in range(1, 7)
            ]
            for name in run_names:
                (artifacts / name).mkdir(parents=True)
            current = artifacts / "20260824T070000Z-00000007"
            current.mkdir()
            unrelated = artifacts / "keep-me"
            unrelated.mkdir()
            linked_run = artifacts / "20260824T000000Z-deadbeef"
            linked_run.symlink_to(unrelated, target_is_directory=True)

            pruned = prune_artifact_runs(
                artifacts,
                current_run=current,
                retention=3,
            )

            self.assertEqual(pruned, run_names[:4])
            self.assertEqual(
                sorted(path.name for path in artifacts.iterdir() if path.is_dir()),
                sorted(
                    [*run_names[4:], current.name, unrelated.name, linked_run.name]
                ),
            )
            self.assertTrue(current.is_dir())
            self.assertTrue(unrelated.is_dir())
            self.assertTrue(linked_run.is_symlink())

    def test_retention_rejects_non_positive_limit(self) -> None:
        from .run import prune_artifact_runs

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            current = artifacts / "20260824T070000Z-00000007"
            current.mkdir(parents=True)

            with self.assertRaisesRegex(
                RigInfrastructureError,
                "E2E_ARTIFACT_RETENTION must be at least 1",
            ):
                prune_artifact_runs(
                    artifacts,
                    current_run=current,
                    retention=0,
                )


class SafetyRailTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_environment_bypasses_system_proxy_for_local_rig(self) -> None:
        from .run import apply_bot_environment

        config = {
            "keytao_base": "http://localhost:3100",
            "bot_token": "local-token",
            "llm": {
                "api_key": "local-llm-key",
                "base_url": "https://api.deepseek.com/",
                "model": "deepseek-v4-flash",
            },
            "bot_values": {},
        }
        with patch.dict(os.environ, {}, clear=True):
            apply_bot_environment(config)
            self.assertEqual(
                os.environ["NO_PROXY"],
                "localhost,127.0.0.1,::1",
            )
            self.assertEqual(os.environ["no_proxy"], os.environ["NO_PROXY"])

    async def test_batch_link_host_guard_enforces_platform_public_host(self) -> None:
        assert_batch_link_hosts(
            "草稿地址：https://keytao.rea.ink/batch/batch-local",
            "qq",
        )
        with self.assertRaisesRegex(
            AssertionError,
            "wrong public host for qq",
        ):
            assert_batch_link_hosts(
                "批次地址：https://keytao.vercel.app/batch/batch-local",
                "qq",
            )
        assert_batch_link_hosts(
            "草稿地址：https://keytao.vercel.app/batch/batch-local",
            "telegram",
        )
        with self.assertRaisesRegex(AssertionError, "wrong public host for telegram"):
            assert_batch_link_hosts(
                "草稿地址：https://keytao.rea.ink/batch/batch-local",
                "telegram",
            )
        with self.assertRaisesRegex(AssertionError, "wrong public host for qq"):
            assert_batch_link_hosts(
                "绑定地址：https://keytao.vercel.app/profile",
                "qq",
            )
        with self.assertRaisesRegex(AssertionError, "wrong public host for telegram"):
            assert_batch_link_hosts(
                "绑定地址：https://keytao.rea.ink/profile",
                "telegram",
            )

    async def test_exact_item_set_ignores_order_but_rejects_duplicates(self) -> None:
        expected = (
            ("Create", "泼天富贵", "ptfg"),
            ("Create", "精神状态", "jeft"),
        )
        self.assertTrue(same_unique_item_set(tuple(reversed(expected)), expected))
        self.assertFalse(
            same_unique_item_set((expected[0], expected[0]), expected)
        )

    async def test_e2e_harness_builds_a_native_onebot_group_reply_segment(
        self,
    ) -> None:
        from keytao_bot.plugins.chat_adapters import extract_onebot_reply_id

        harness = object.__new__(E2EBotHarness)
        harness._sent_messages = {501: {"message_id": 501}}
        harness.message_timeout = 1.0
        harness.replies = []
        harness.reply_event = asyncio.Event()
        harness._current_event = ContextVar("test_e2e_current_event", default=None)
        harness.recorder = MagicMock()

        class FakeBot:
            self_id = "99999999999999999999999999999999"

        class FakeOpenAIChat:
            captured_event = None

            async def should_handle(self, _bot, _event):
                return True

            async def handle_ai_chat(self, _bot, event):
                self.captured_event = event
                harness.replies.append("accepted")
                harness.reply_event.set()

        harness.bot = FakeBot()
        harness.openai_chat = FakeOpenAIChat()

        reply = await harness.send_group_reply(
            platform_id="123456789012345678901234567890",
            sender_name="Rea",
            text="都加",
            reply_message_id=501,
            to_me=True,
        )

        event = harness.openai_chat.captured_event
        self.assertEqual(reply, "accepted")
        self.assertEqual(extract_onebot_reply_id(event), "501")
        self.assertEqual(event.get_plaintext(), "都加")
        self.assertEqual(
            harness.recorder.record_message.call_args.kwargs["reply_message_id"],
            501,
        )

    async def test_local_next_client_retries_transport_errors_with_fresh_pool(
        self,
    ) -> None:
        request = httpx.Request(
            "GET",
            "http://localhost:3100/api/phrases/encode?word=亮面",
        )

        class FakeAsyncClient:
            def __init__(self, *results: Any) -> None:
                self.request = AsyncMock(side_effect=results)
                self.aclose = AsyncMock()

            async def __aenter__(self) -> "FakeAsyncClient":
                return self

            async def __aexit__(self, *_args: Any) -> None:
                await self.aclose()

        clients = [
            FakeAsyncClient(
                httpx.ConnectTimeout("connect busy", request=request),
            ),
            FakeAsyncClient(
                httpx.ReadTimeout("route busy", request=request),
            ),
            FakeAsyncClient(
                httpx.ConnectError("connection reset", request=request),
            ),
            FakeAsyncClient(
                httpx.Response(200, json={"word": "亮面"}, request=request),
                httpx.Response(200, json={"word": "亮面"}, request=request),
            ),
        ]
        client = LocalNextClient(
            base_url="http://localhost:3100",
            bot_token="test",
        )

        with (
            patch("e2e.runtime.httpx.AsyncClient", side_effect=clients) as factory,
            patch("e2e.runtime.asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
        ):
            first = await client.encode("亮面")
            second = await client.encode("亮面")
            await client.close()

        self.assertEqual(first, {"word": "亮面"})
        self.assertEqual(second, {"word": "亮面"})
        self.assertEqual(factory.call_count, 4)
        configured_timeout = factory.call_args_list[0].kwargs["timeout"]
        self.assertEqual(configured_timeout.connect, 5.0)
        self.assertEqual(configured_timeout.pool, 5.0)
        self.assertEqual(configured_timeout.read, 90.0)
        self.assertEqual(
            sleep_mock.await_args_list,
            [call(0.5), call(1.0), call(2.0)],
        )
        for failed_client in clients[:3]:
            failed_client.aclose.assert_awaited_once_with()
        self.assertEqual(clients[3].request.await_count, 2)
        clients[3].aclose.assert_awaited_once_with()

    def test_local_socket_stats_count_time_wait_for_next_port(self) -> None:
        netstat = """\
tcp4  0  0  127.0.0.1.49152  127.0.0.1.3100  TIME_WAIT
tcp4  0  0  127.0.0.1.3100   127.0.0.1.49153 TIME_WAIT
tcp4  0  0  127.0.0.1.49154  127.0.0.1.9999  TIME_WAIT
tcp4  0  0  127.0.0.1.3100   127.0.0.1.49155 ESTABLISHED
"""
        completed = subprocess.CompletedProcess(
            args=["netstat", "-an", "-p", "tcp"],
            returncode=0,
            stdout=netstat,
            stderr="",
        )
        with patch("e2e.run.subprocess.run", return_value=completed):
            stats = collect_local_socket_stats("http://localhost:3100")

        self.assertEqual(stats["status"], "captured")
        self.assertEqual(stats["tcpTimeWaitCount"], 3)
        self.assertEqual(stats["targetPortTimeWaitCount"], 2)

    def test_abort_record_keeps_wrapped_transport_request_target(self) -> None:
        request = httpx.Request(
            "GET",
            "http://localhost:3100/api/phrases/encode?word=亮面",
        )
        try:
            try:
                raise httpx.ConnectTimeout("", request=request)
            except httpx.ConnectTimeout as transport_error:
                raise RigInfrastructureError("warm-up exhausted") from transport_error
        except RigInfrastructureError as error:
            record = abort_record_for_error(error)

        self.assertEqual(record["type"], "RigInfrastructureError")
        self.assertEqual(record["transportErrorType"], "ConnectTimeout")
        self.assertEqual(
            record["request"],
            {
                "method": "GET",
                "url": "http://localhost:3100/api/phrases/encode?word=%E4%BA%AE%E9%9D%A2",
            },
        )

    def test_review_source_domains_are_explicitly_blocked(self) -> None:
        expected = {
            "baike.baidu.com",
            "cd.hwxnet.com",
            "cidian.qianp.com",
            "moedict.tw",
            "www.moedict.tw",
            "www.zdic.net",
            "zd.hwxnet.com",
            "zdic.net",
            "zh.wikipedia.org",
        }
        self.assertEqual(BLOCKED_EXTERNAL_DOMAINS, expected)
        with patch.object(
            NetworkAllowlist,
            "_resolve_llm_ips",
            return_value=frozenset({"203.0.113.10"}),
        ):
            guard = NetworkAllowlist(llm_base_url="https://llm.example.com/v1")
        for domain in ("cd.hwxnet.com", "zd.hwxnet.com"):
            with self.subTest(domain=domain):
                with self.assertRaisesRegex(
                    SafetyViolation,
                    "Blocked external review domain",
                ):
                    guard.assert_url_allowed(f"https://{domain}/search.do?wd=test")

    def test_weight_rule_prompt_copy_has_one_canonical_statement(self) -> None:
        skill = (
            Path(__file__).parents[1]
            / "keytao_bot/skills/keytao-draft/SKILL.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(skill.count("权重规则（唯一口径）"), 1)
        self.assertIn("数值越小排序越靠前", skill)
        self.assertIn("Link=10000", skill)
        self.assertNotIn("（权重 0）", skill)

    def test_advertised_chain_reorder_phrasing_replays_through_closed_parser(self) -> None:
        from keytao_bot.plugins.chat_routing import _parse_code_chain_reorder_command

        lookup_skill = (
            Path(__file__).parents[1]
            / "keytao_bot/skills/keytao-lookup/SKILL.md"
        ).read_text(encoding="utf-8")
        advertised = (
            "重新排序 mkdr 编码链按常用度",
            "把 mkdr 这条链按常用度排一下",
            "重排 mkdr",
        )
        for phrase in advertised:
            with self.subTest(phrase=phrase):
                self.assertIn(f"`{phrase}`", lookup_skill)
                parsed = _parse_code_chain_reorder_command(phrase)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.code, "mkdr")

    def test_user_visible_reply_guard_rejects_internal_pr_identifiers(self) -> None:
        from keytao_bot.plugins.chat_render import _assert_plain_user_facing_reply

        with self.assertRaises(ValueError):
            _assert_plain_user_facing_reply(
                "服务端已锁定删除目标：PR#3062 呵呵呵 @ hhhooo"
            )

    def test_e2e_docs_describe_repaired_and_new_scenarios(self) -> None:
        readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
        self.assertIn("S11 keeps the confirmation path", readme)
        self.assertIn("rejects provisional batch links", readme)
        self.assertIn("S12 asserts that narrow", readme)
        self.assertIn("exact persisted weight cascade", readme)
        self.assertIn("S13 sends an explicit weight", readme)
        self.assertIn("asks the user to resend the same current message", readme)
        self.assertIn("S14 injects a 汉典-shaped search hit", readme)
        self.assertIn("S15 first discovers", readme)
        self.assertIn("S16 replays the two-word 载流", readme)
        self.assertIn("S17 exercises the common-characters-plus-LLM", readme)
        self.assertIn("S18 replays the multi-number candidate incident", readme)
        self.assertIn("S19 replays the oversized advertised-set incident", readme)
        self.assertIn("S20 replays native-quoted batch assent", readme)
        self.assertIn("S21 replays the 2026-08-16 advertised-contract incidents", readme)
        self.assertIn("S22 replays the orphaned re-review advertisement incident", readme)
        self.assertIn("S23 replays the stale advertised-assent production incident", readme)
        self.assertIn("S24 replays the single-word natural-assent incident", readme)
        self.assertIn("S25 replays the 炒冷饭 production incident", readme)
        self.assertIn("S26 replays the add-plus-eviction incident", readme)
        self.assertIn("S37 sends the verbatim occupant-derived eviction", readme)
        self.assertIn("S27 replays the binding-precheck incident", readme)
        self.assertIn("S28 replays the multi-reading candidate cascade", readme)
        self.assertIn("S29 replays the 2026-08-20 quoted-summary incident", readme)
        self.assertIn("S30 pins three intent-coverage boundaries", readme)
        self.assertIn("S31 executes the verbatim incident command", readme)
        self.assertIn("S32 replays both 2026-08-20 chain-scope incidents", readme)
        self.assertIn("S33 replays both 2026-08-21 homophone batch shapes", readme)
        self.assertIn("S34 replays the 2026-08-21 pending-batch incident", readme)
        self.assertIn("S35 replays the 2026-08-22 default-reorder incident", readme)
        self.assertIn("S36 replays the 2026-08-23 delete-and-swap incident round", readme)
        self.assertIn("S39 collapses the explicit-reading eviction flow", readme)
        self.assertIn("S40 closes the 2026-08-25 assent-execution incident round", readme)
        self.assertIn("S41 closes the 2026-08-26 reading-reply duplication incident", readme)
        self.assertIn("S42 closes the 2026-08-26 missing-affordance incident", readme)
        self.assertIn("S43 closes the 2026-08-28 encode-timeout incident", readme)
        self.assertIn("S44 closes the 2026-08-29 compound-selection incident", readme)
        self.assertIn("S45 closes the 2026-08-30 swap-verb", readme)
        self.assertIn("S46 closes the 2026-08-31 promise-preserving", readme)
        self.assertIn("S47 closes the 2026-09-01 choice-closure incident", readme)
        self.assertIn(
            "S48 closes the 2026-09-03 numbered-candidate binding incident",
            readme,
        )
        self.assertIn(
            "S50 replays the 2026-09-04 relative-position incident",
            readme,
        )
        self.assertIn(
            "whole-word `corpus_frequency` and `common_characters_and_llm` routes",
            readme,
        )

    def test_s45_pins_swap_and_character_question_contracts(self) -> None:
        self.assertEqual(S45_FIRST_WORD, "财宝")
        self.assertEqual(S45_SECOND_WORD, "财报")
        self.assertEqual(S45_FIRST_CODE, "chbz")
        self.assertEqual(S45_SECOND_CODE, "chbza")
        self.assertEqual(S45_SWAP_MESSAGE, "对换财宝和财报的编码")
        self.assertEqual(S45_CHARACTER_QUESTION, "单人旁加个巨字是什么字")
        self.assertEqual(S45_CHARACTER_ANSWER, "佢")
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S45"]
        self.assertEqual(fixture["probe_words"], ("财宝", "财报", "佢"))
        rows_by_key = {
            (row["kind"], row["entry"]): row
            for row in fixture["rows"]
        }
        self.assertEqual(rows_by_key[("entry", "财宝")]["pinyins"], ["cái", "bǎo"])
        self.assertEqual(rows_by_key[("entry", "财报")]["pinyins"], ["cái", "bào"])
        self.assertEqual(rows_by_key[("char", "佢")]["pinyins"], ["qú"])

    def test_s46_pins_promise_preserving_double_eviction_contract(self) -> None:
        self.assertEqual(S46_WORD, "哲思")
        self.assertEqual(S46_OCCUPANT, "这厮")
        self.assertEqual(S46_FIRST_CODE, "fesk")
        self.assertEqual(S46_FIRST_SHIFTED_CODE, "fesko")
        self.assertEqual(S46_SECOND_CODE, "qesk")
        self.assertEqual(S46_SECOND_SHIFTED_CODE, "qesko")
        self.assertEqual(
            S46_MESSAGE,
            '加词 哲思 fesk，并且为"这厮 fesk"顺延\n'
            '加词 哲思 qesk，并且为"这厮 qesk"顺延',
        )
        self.assertEqual(
            S46_PLAN_COMMAND,
            "添加「哲思」 fesk，这厮顺延\n添加「哲思」 qesk，这厮顺延",
        )
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S46"]
        self.assertEqual(fixture["probe_words"], (S46_WORD, S46_OCCUPANT))
        entries = {
            row["entry"]: row["pinyins"]
            for row in fixture["rows"]
            if row["kind"] == "entry"
        }
        self.assertEqual(entries[S46_WORD], ["zhé", "sī"])
        self.assertEqual(entries[S46_OCCUPANT], ["zhè", "sī"])

    def test_scenario_pack_is_contiguous_through_s50(self) -> None:
        self.assertEqual(
            [scenario.scenario_id for scenario in SCENARIOS],
            [f"S{index}" for index in range(1, 51)],
        )

    def test_s48_pins_numbered_create_with_eviction_contract(self) -> None:
        self.assertEqual((S48_WORD, S48_OCCUPANT), ("单份", "蛋粉"))
        self.assertEqual(S48_TARGET_CODE, "dffn")
        self.assertEqual(S48_RECOMMENDED_CODE, "dffno")
        self.assertEqual(S48_SHIFTED_CODE, "dffna")
        self.assertEqual(S48_NUMBERED_EVICTION, "加入1，挤掉蛋粉")
        self.assertEqual(S48_NUMBERED_RECODE, "添加1，并为蛋粉重新编码")
        self.assertEqual(S48_EXPLICIT_EVICTION, "添加 单份 dffn，挤掉蛋粉")
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S48"]
        self.assertEqual(fixture["probe_words"], (S48_WORD, S48_OCCUPANT))
        entries = {
            row["entry"]: row["pinyins"]
            for row in fixture["rows"]
            if row["kind"] == "entry"
        }
        self.assertEqual(entries[S48_WORD], ["dān", "fèn"])
        self.assertEqual(entries[S48_OCCUPANT], ["dàn", "fěn"])

    def test_s50_pins_relative_position_incident_transcript(self) -> None:
        self.assertEqual(S50_DISCOVERY, "喵喵 小像")
        self.assertEqual(
            S50_CONTEXT,
            "小像确实比较常用，属于美团超市这块，换到前面",
        )
        self.assertEqual(S50_FRONT, "把 小像 放在 销项 前面")
        self.assertEqual(
            S50_COMPOSITE_FRONT,
            "把小象放在销项前面，顺延后面的词",
        )
        self.assertEqual(S50_CORRECTION, "错了 是小象")
        self.assertEqual(S50_AFTER, "小象在肖像后面")
        self.assertEqual(S50_INITIAL_FREE_CODE, "xcxxii")
        self.assertEqual(
            (
                S50_INITIAL_WORD,
                S50_REPLACEMENT_WORD,
                S50_DESTINATION,
                S50_OCCUPANT,
            ),
            ("小像", "小象", "肖像", "小箱"),
        )
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S50"]
        self.assertEqual(
            fixture["probe_words"],
            ("小像", "小象", "销项", "肖像", "小箱"),
        )

    def test_s37_declares_owned_eviction_fixture_readings(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S37"]
        self.assertEqual(fixture["probe_words"], (S37_WORD, S37_OCCUPANT))
        entries = {
            row["entry"]: row["pinyins"]
            for row in fixture["rows"]
            if row["kind"] == "entry"
        }
        self.assertEqual(entries[S37_WORD], ["pá", "pá", "gān"])
        self.assertEqual(entries[S37_OCCUPANT], ["pí", "pá", "gǔ"])

    def test_s38_pins_the_incident_messages_and_fixture(self) -> None:
        self.assertEqual(S38_EXPLICIT_READING_MESSAGE, "加词 出圈，读音是 chū quān")
        self.assertEqual(
            S38_EXPLANATION_MESSAGE,
            "耙耙柑为pá pá gān，因此这三个字的声母分别为p, p, g",
        )
        self.assertEqual(
            S38_NEGATIVE_MODIFIER_MESSAGE,
            "加词 耙耙柑 ppg，不要顺延其他相关的词条",
        )
        self.assertEqual(S38_QUERY_CONTROLS, ("1", "回复1", "加入"))
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S38"]
        self.assertEqual(fixture["probe_words"], ("出圈", S37_WORD, S37_OCCUPANT))
        entries = {
            row["entry"]: (row["status"], row["pinyins"])
            for row in fixture["rows"]
            if row["kind"] == "entry"
        }
        self.assertEqual(entries["出圈"], ("absent", []))
        out_circle = next(
            row
            for row in fixture["rows"]
            if row["kind"] == "entry" and row["entry"] == "出圈"
        )
        self.assertEqual(
            out_circle["expected_pronunciation_source"],
            "zdic-character-default",
        )
        self.assertTrue(out_circle["expected_semantic_pronunciation_needed"])
        self.assertEqual(out_circle["expected_context_pinyins"], ["chū", "quān"])

    def test_s39_pins_reading_selection_and_occupant_fixture(self) -> None:
        self.assertEqual(S39_COMMAND, "加词 出圈 圈字读quan")
        self.assertEqual(S39_SELECTION, "1 重新编码")
        self.assertEqual((S39_WORD, S39_OCCUPANT, S39_TARGET_CODE), (
            "出圈", "除权", "jjqt",
        ))
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S39"]
        self.assertEqual(fixture["probe_words"], (S39_WORD, S39_OCCUPANT))
        entries = {
            row["entry"]: (row["status"], row["pinyins"])
            for row in fixture["rows"]
            if row["kind"] == "entry"
        }
        self.assertEqual(entries[S39_WORD], ("absent", []))
        self.assertEqual(entries[S39_OCCUPANT], ("found", ["chú", "quán"]))

    def test_s40_declares_combined_incident_fixture(self) -> None:
        self.assertEqual(
            (S40_COPY_WORD, S40_OCCUPANT, S40_TARGET_CODE),
            ("发布会", "重病号", "fbh"),
        )
        self.assertEqual(S40_BATCH_WORDS, S23_BATCH_WORDS[:2])
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S40"]
        self.assertEqual(
            fixture["probe_words"],
            (*ZDIC_FIXTURES_BY_SCENARIO["S35"]["probe_words"], *S40_BATCH_WORDS),
        )

    def test_s41_pins_reading_and_code_questions(self) -> None:
        self.assertEqual(S41_WORD, "畜产品")
        self.assertEqual(S41_READING_MESSAGE, "畜产品的畜字怎么读")
        self.assertEqual(S41_CODE_MESSAGE, "畜产品怎么编码")
        self.assertEqual(S41_EXISTING_CODES, ("xjpoo", "jjpoo"))

    def test_s42_pins_the_three_word_affordance_incident(self) -> None:
        self.assertEqual(S42_WORDS, ("老登", "中登", "小登"))
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S42"]
        self.assertEqual(fixture["probe_words"], S42_WORDS)
        entries = {
            row["entry"]: (row["status"], row["pinyins"])
            for row in fixture["rows"]
            if row["kind"] == "entry"
        }
        self.assertEqual(
            entries,
            {word: ("absent", []) for word in S42_WORDS},
        )

    def test_s43_pins_encode_timeout_recovery_and_degradation(self) -> None:
        self.assertEqual(S43_WORD, "钉选")
        self.assertIn("S43", {scenario.scenario_id for scenario in SCENARIOS})
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S43"]
        self.assertEqual(fixture["probe_words"], (S43_WORD,))
        controller = EncodeDelayController(0.20, 0.05)
        controller.arm("S43", injections=3)
        decisions = [
            controller.should_inject(
                scenario_id="S43",
                method="GET",
                path="/api/phrases/encode",
            )
            for _ in range(4)
        ]
        self.assertEqual(decisions, [True, True, True, False])
        self.assertEqual(controller.injection_count, 3)

    def test_s44_pins_compound_selection_and_owned_fixture(self) -> None:
        from . import scenarios as scenario_module

        self.assertEqual(scenario_module.S44_WORD, "载具")
        self.assertEqual(scenario_module.S44_OCCUPANT, "在距")
        self.assertEqual(scenario_module.S44_OCCUPIED_CODE, "zhjl")
        self.assertEqual(scenario_module.S44_FREE_CODE, "zhjlu")
        self.assertEqual(
            scenario_module.S44_COMMAND,
            "1 重新编码，并加入 2",
        )
        self.assertEqual(scenario_module.S44_DISCOVERY, "加词 载具")
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S44"]
        self.assertEqual(
            fixture["probe_words"],
            (scenario_module.S44_WORD, scenario_module.S44_OCCUPANT),
        )
        characters = {
            row["entry"]: row["pinyins"]
            for row in fixture["rows"]
            if row["kind"] == "char"
        }
        self.assertEqual(characters["载"], ["zǎi", "zài"])
        rows_by_key = {
            (row["kind"], row["entry"]): row
            for row in fixture["rows"]
        }
        self.assertTrue(_encoded_matches_zdic_fixture(
            word="载具",
            encoded={
                "pronunciationSource": "zdic-phrase",
                "standardPronunciationStatus": "found",
                "semanticPronunciationNeeded": False,
                "chars": [
                    {
                        "char": "载",
                        "pinyin": "zài",
                        "pinyins": ["zài", "zǎi"],
                        "pronunciationLookupStatus": "found",
                    },
                    {
                        "char": "具",
                        "pinyin": "jù",
                        "pinyins": ["jù"],
                        "pronunciationLookupStatus": "found",
                    },
                ],
            },
            rows_by_key=rows_by_key,
        ))

    def test_all_scenario_zdic_declarations_merge_without_conflicts(self) -> None:
        scenario_ids = tuple(scenario.scenario_id for scenario in SCENARIOS)
        rows = zdic_cache_rows_for_scenarios(scenario_ids)
        self.assertTrue(rows)
        self.assertEqual(
            sum(
                row["kind"] == "char" and row["entry"] == "载"
                for row in rows
            ),
            1,
        )

    def test_recommended_empty_code_accepts_executable_opt_out(self) -> None:
        reply = (
            "候选编码:\n"
            "1. wkxk — 已有「赤溪」 ← 常用度推荐（需重排）\n"
            "2. wkxko — 空位\n"
            "推荐：\n"
            "- “「吃席」占 wkxk、「赤溪」顺延”（吃席、赤溪）\n"
            "不重排选 2（wkxko）。"
        )
        self.assertEqual(
            _recommended_empty_code(reply, word="吃席"),
            "wkxko",
        )

    def test_s35_declares_isolated_reorder_and_free_slot_controls(self) -> None:
        self.assertEqual(
            S35_FRONT_CASES,
            (
                ("发布会", "重病号", "fbh"),
                ("计算机", "建三江", "jsj"),
            ),
        )
        self.assertEqual(S35_FREE_CONTROL, ("无事忙", "wem"))
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S35"]
        self.assertEqual(
            fixture["probe_words"],
            ("发布会", "重病号", "计算机", "建三江", "无事忙"),
        )

    def test_artifacts_redact_admin_credentials(self) -> None:
        payload = _redact_sensitive(
            {
                "name": "reserved-admin",
                "password": "secret-password",
                "nested": {"token": "signed-jwt"},
            }
        )
        self.assertEqual(payload["name"], "reserved-admin")
        self.assertEqual(payload["password"], "[REDACTED]")
        self.assertEqual(payload["nested"]["token"], "[REDACTED]")

    def test_keytao_base_requires_localhost(self) -> None:
        self.assertEqual(
            validate_keytao_base("http://localhost:3100"),
            "http://localhost:3100",
        )
        with self.assertRaises(SafetyViolation):
            validate_keytao_base("https://keytao.vercel.app")
        with self.assertRaises(SafetyViolation):
            validate_keytao_base("http://192.0.2.2:3100")

    def test_next_database_must_resolve_local(self) -> None:
        result = validate_next_database_url(
            "postgresql://dev:dev@localhost:5432/keytao"
        )
        self.assertEqual(result["host"], "localhost")
        with self.assertRaises(SafetyViolation):
            validate_next_database_url(
                "postgresql://user:pass@db.production.example:5432/keytao"
            )

    @patch("e2e.zdic_seed.subprocess.run")
    def test_zdic_seeder_refuses_non_localhost_database(self, run_mock) -> None:
        with self.assertRaises(SafetyViolation):
            seed_s9_zdic_cache(
                "postgresql://user:pass@db.production.example:5432/keytao",
                next_dir=Path("/not-inspected-before-database-validation"),
            )
        run_mock.assert_not_called()

    @patch("e2e.zdic_seed.subprocess.run")
    def test_multi_add_zdic_seed_is_scenario_driven_and_deduplicated(
        self,
        run_mock,
    ) -> None:
        expected_rows = {
            ("char", "王", "found", ("wáng",)),
            ("char", "中", "found", ("zhōng",)),
            ("char", "微", "found", ("wēi",)),
            ("char", "服", "found", ("fú",)),
            ("char", "务", "found", ("wù",)),
            ("entry", "王中王", "absent", ()),
            ("entry", "微服务", "absent", ()),
        }
        for scenario_id in ("S10",):
            declared = ZDIC_FIXTURES_BY_SCENARIO[scenario_id]
            self.assertEqual(set(declared["probe_words"]), {"王中王", "微服务"})
            self.assertEqual(
                {
                    (
                        row["kind"],
                        row["entry"],
                        row["status"],
                        tuple(row["pinyins"]),
                    )
                    for row in declared["rows"]
                },
                expected_rows,
            )

        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = ""
        run_mock.return_value.stderr = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            next_dir = Path(temp_dir)
            prisma = next_dir / "node_modules" / ".bin" / "prisma"
            prisma.parent.mkdir(parents=True)
            prisma.touch()
            (next_dir / "prisma.config.ts").touch()
            result = seed_zdic_cache(
                "postgresql://dev:dev@localhost:5432/keytao",
                next_dir=next_dir,
                scenario_ids=("S10",),
            )

        self.assertEqual(result["scenarioIds"], ["S10"])
        self.assertEqual(
            {
                (
                    row["kind"],
                    row["entry"],
                    row["status"],
                    tuple(row["pinyins"]),
                )
                for row in result["rows"]
            },
            expected_rows,
        )
        self.assertEqual(len(result["rows"]), len(expected_rows))
        seed_sql = run_mock.call_args.kwargs["input"]
        for _kind, entry, _status, _pinyins in expected_rows:
            self.assertEqual(seed_sql.count(f"'{entry}'"), 1)

    def test_s14_zdic_seed_declares_exact_character_readings(self) -> None:
        self.assertIn("S14", {scenario.scenario_id for scenario in SCENARIOS})
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S14"]
        self.assertEqual(fixture["probe_words"], ("亮面",))
        self.assertEqual(
            {
                (
                    row["kind"],
                    row["entry"],
                    row["status"],
                    tuple(row["pinyins"]),
                )
                for row in fixture["rows"]
            },
            {
                ("char", "亮", "found", ("liàng",)),
                ("char", "面", "found", ("miàn",)),
                ("entry", "亮面", "absent", ()),
            },
        )

    def test_s15_reuses_the_s9_and_s14_zdic_fixtures(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S15"]
        self.assertEqual(fixture["probe_words"], ("射覆", "亮面"))
        expected = {
            (row["kind"], row["entry"], row["status"], tuple(row["pinyins"]))
            for scenario_id in ("S9", "S14")
            for row in ZDIC_FIXTURES_BY_SCENARIO[scenario_id]["rows"]
        }
        actual = {
            (row["kind"], row["entry"], row["status"], tuple(row["pinyins"]))
            for row in fixture["rows"]
        }
        self.assertEqual(actual, expected)

    def test_s16_zdic_seed_declares_dictionary_occupant_readings(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S16"]
        self.assertEqual(fixture["probe_words"], ("载流", "载流子", "座落在"))
        declared = {
            (row["kind"], row["entry"]): (row["status"], tuple(row["pinyins"]))
            for row in fixture["rows"]
        }
        self.assertEqual(declared[("char", "座")], ("found", ("zuò",)))
        self.assertEqual(declared[("char", "落")], ("found", ("luò",)))
        self.assertEqual(declared[("char", "在")], ("found", ("zài",)))
        self.assertEqual(declared[("entry", "座落在")], ("absent", ()))

    def test_s17_zdic_seed_exercises_absent_words_with_known_characters(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S17"]
        self.assertEqual(fixture["probe_words"], ("产季", "龘季"))
        declared = {
            (row["kind"], row["entry"]): (row["status"], tuple(row["pinyins"]))
            for row in fixture["rows"]
        }
        self.assertEqual(declared[("char", "产")], ("found", ("chǎn",)))
        self.assertEqual(declared[("char", "季")], ("found", ("jì",)))
        self.assertEqual(declared[("char", "龘")], ("found", ("dá",)))
        self.assertEqual(declared[("entry", "产季")], ("absent", ()))
        self.assertEqual(declared[("entry", "龘季")], ("absent", ()))

    def test_s18_zdic_seed_declares_multi_reading_candidate_reality(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S18"]
        self.assertEqual(fixture["probe_words"], ("还车", "换车"))
        declared = {
            (row["kind"], row["entry"]): (row["status"], tuple(row["pinyins"]))
            for row in fixture["rows"]
        }
        self.assertEqual(declared[("char", "还")], ("found", ("huán", "hái")))
        self.assertEqual(declared[("char", "车")], ("found", ("chē",)))
        self.assertEqual(declared[("char", "换")], ("found", ("huàn",)))
        self.assertEqual(
            declared[("entry", "还车")],
            ("found", ("huán", "chē")),
        )
        self.assertEqual(declared[("entry", "换车")], ("absent", ()))

    def test_s19_zdic_seed_declares_the_complete_advertised_word_set(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S19"]
        self.assertEqual(len(fixture["probe_words"]), 11)
        expected_character_pinyins = {
            "显": ("xiǎn",),
            "眼": ("yǎn",),
            "包": ("bāo",),
            "嘴": ("zuǐ",),
            "替": ("tì",),
            "松": ("sōng",),
            "弛": ("chí",),
            "感": ("gǎn",),
            "电": ("diàn",),
            "子": ("zǐ",),
            "榨": ("zhà",),
            "菜": ("cài",),
            "情": ("qíng",),
            "绪": ("xù",),
            "价": ("jià",),
            "值": ("zhí",),
            "班": ("bān",),
            "味": ("wèi",),
            "泼": ("pō",),
            "天": ("tiān",),
            "富": ("fù",),
            "贵": ("guì",),
            "精": ("jīng",),
            "神": ("shén",),
            "状": ("zhuàng",),
            "态": ("tài",),
            "职": ("zhí",),
            "场": ("chǎng",),
            "搭": ("dā",),
            "选": ("xuǎn",),
            "打": ("dǎ",),
            "工": ("gōng",),
            "人": ("rén",),
            "沙": ("shā",),
            "县": ("xiàn",),
            "小": ("xiǎo",),
            "吃": ("chī",),
        }
        self.assertEqual(
            set(expected_character_pinyins),
            set("".join(fixture["probe_words"])),
        )
        self.assertEqual(
            {
                row["entry"]: (row["status"], tuple(row["pinyins"]))
                for row in fixture["rows"]
                if row["kind"] == "char"
            },
            {
                char: ("found", pinyins)
                for char, pinyins in expected_character_pinyins.items()
            },
        )
        self.assertEqual(
            {
                row["entry"]: (row["status"], tuple(row["pinyins"]))
                for row in fixture["rows"]
                if row["kind"] == "entry"
            },
            {
                "显眼包": ("found", ("xiǎn", "yǎn", "bāo")),
                "嘴替": ("found", ("zuǐ", "tì")),
                "松弛感": ("found", ("sōng", "chí", "gǎn")),
                "电子榨菜": ("found", ("diàn", "zǐ", "zhà", "cài")),
                "情绪价值": ("found", ("qíng", "xù", "jià", "zhí")),
                "班味": ("found", ("bān", "wèi")),
                "泼天富贵": ("found", ("pō", "tiān", "fù", "guì")),
                "精神状态": ("found", ("jīng", "shén", "zhuàng", "tài")),
                "职场搭子": ("found", ("zhí", "chǎng", "dā", "zǐ")),
                "天选打工人": ("found", ("tiān", "xuǎn", "dǎ", "gōng", "rén")),
                "沙县小吃": ("found", ("shā", "xiàn", "xiǎo", "chī")),
            },
        )

    def test_s20_reuses_the_exact_three_word_s19_fixture_shape(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S20"]
        self.assertEqual(fixture["probe_words"], S20_BATCH_WORDS)
        self.assertEqual(
            {
                row["entry"]
                for row in fixture["rows"]
                if row["kind"] == "entry"
            },
            set(S20_BATCH_WORDS),
        )
        self.assertEqual(
            {
                row["entry"]
                for row in fixture["rows"]
                if row["kind"] == "char"
            },
            set("".join(S20_BATCH_WORDS)),
        )

    def test_s21_zdic_seed_declares_the_minimal_contract_word_set(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S21"]
        self.assertEqual(fixture["probe_words"], S21_BATCH_WORDS)
        rows = {
            (row["kind"], row["entry"]): tuple(row["pinyins"])
            for row in fixture["rows"]
        }
        self.assertEqual(
            {entry for kind, entry in rows if kind == "char"},
            set("".join(S21_BATCH_WORDS)),
        )
        self.assertEqual(
            {entry for kind, entry in rows if kind == "entry"},
            set(S21_BATCH_WORDS),
        )
        self.assertEqual(rows[("entry", "嘴替")], ("zuǐ", "tì"))

    def test_s22_uses_the_minimum_two_word_incident_fixture(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S22"]
        self.assertEqual(fixture["probe_words"], S22_BATCH_WORDS)
        self.assertEqual(S22_BATCH_WORDS, S19_ADVERTISED_WORDS[:2])
        self.assertEqual(
            {
                row["entry"]
                for row in fixture["rows"]
                if row["kind"] == "entry"
            },
            set(S22_BATCH_WORDS),
        )
        self.assertEqual(
            {
                row["entry"]
                for row in fixture["rows"]
                if row["kind"] == "char"
            },
            set("".join(S22_BATCH_WORDS)),
        )

    def test_s23_reuses_the_exact_nine_word_incident_fixture(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S23"]
        self.assertEqual(S23_BATCH_WORDS, S19_ADVERTISED_WORDS[:9])
        self.assertEqual(fixture["probe_words"], S23_BATCH_WORDS)
        self.assertEqual(
            {row["entry"] for row in fixture["rows"] if row["kind"] == "entry"},
            set(S23_BATCH_WORDS),
        )

    def test_s24_reuses_the_s18_single_word_fixture(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S24"]
        self.assertEqual(fixture, ZDIC_FIXTURES_BY_SCENARIO["S18"])
        self.assertIn(S24_WORD, fixture["probe_words"])

    def test_s25_owns_the_exact_flykey_incident_fixture(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S25"]
        self.assertEqual(fixture["probe_words"], (S25_WORD,))
        self.assertEqual(
            {row["entry"] for row in fixture["rows"] if row["kind"] == "char"},
            set(S25_WORD),
        )
        whole_word = next(
            row
            for row in fixture["rows"]
            if row["kind"] == "entry" and row["entry"] == S25_WORD
        )
        self.assertEqual(whole_word["pinyins"], ["chǎo", "lěng", "fàn"])

    def test_s27_owns_the_exact_binding_precheck_word_fixture(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S27"]
        self.assertEqual(fixture["probe_words"], (S27_WORD,))
        self.assertEqual(
            {row["entry"] for row in fixture["rows"] if row["kind"] == "char"},
            set(S27_WORD),
        )
        whole_word = next(
            row
            for row in fixture["rows"]
            if row["kind"] == "entry" and row["entry"] == S27_WORD
        )
        self.assertEqual(whole_word["pinyins"], ["lái", "dōu", "lái", "le"])

    def test_s28_reuses_the_seeded_multi_reading_fixture(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S28"]
        self.assertEqual(fixture, ZDIC_FIXTURES_BY_SCENARIO["S18"])
        self.assertEqual(S28_WORD, "还车")
        self.assertEqual(S28_INVALID_CODE, "zzzzzz")
        self.assertIn("S28", {scenario.scenario_id for scenario in SCENARIOS})

    def test_s30_reuses_the_absent_whole_word_shape_for_e2e_deduplication(self) -> None:
        self.assertEqual(
            ZDIC_FIXTURES_BY_SCENARIO["S30"],
            ZDIC_FIXTURES_BY_SCENARIO["S2"],
        )

    def test_s32_owns_both_verbatim_chain_scope_incidents(self) -> None:
        self.assertEqual(
            S32_CHAIN_COMMAND,
            "重新排序下mkdr 编码链这几个词按优先级",
        )
        self.assertEqual(
            S32_WORD_LIST_COMMAND,
            "重新排序下\n米等\n幂等\n迷瞪",
        )
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S32"]
        self.assertEqual(fixture["probe_words"], ("米等", "幂等", "迷瞪"))
        declared = {
            (row["kind"], row["entry"]): (row["status"], tuple(row["pinyins"]))
            for row in fixture["rows"]
        }
        self.assertEqual(declared[("entry", "米等")], ("found", ("mǐ", "děng")))
        self.assertEqual(declared[("entry", "幂等")], ("found", ("mì", "děng")))
        self.assertEqual(declared[("entry", "迷瞪")], ("found", ("mí", "dèng")))

    def test_s33_owns_the_exact_homophone_incident_fixture(self) -> None:
        self.assertEqual(S33_WORDS, ("洒漏", "撒漏"))
        self.assertEqual(S33_SIX_WORDS, ("洒漏", "洒溇"))
        self.assertEqual(S33_DISCOVERY, "喵喵 加词 洒漏 撒漏")
        self.assertEqual(S33_EXTERNAL_WORDS, ("缩手", "所售"))
        self.assertEqual(S33_EXTERNAL_OCCUPANT, ("所受", "sled"))
        self.assertEqual(S33_EXTERNAL_QUERY, "缩手 所售")
        self.assertEqual(
            S33_EXTERNAL_EXPECTED,
            (("缩手", "sleda"), ("所售", "sledu")),
        )
        self.assertEqual(
            _s33_external_query_pairs(
                "缩手（suō shǒu）\n"
                "推荐编码：sleda\n"
                "1. sled — 已有「所受」\n"
                "2. sleda — 空位 ✅（推荐）\n"
                "所售（suǒ shòu）\n"
                "1. sled — 已有「所受」\n"
                "2. sledu — 空位 ✅\n"
            ),
            S33_EXTERNAL_EXPECTED,
        )
        self.assertEqual(
            _s33_external_query_pairs(
                "缩手（suō shǒu）\n1. sleda — 空位\n"
                "所售（suǒ shòu）\n1. sledu — 空位\n"
            ),
            (),
        )
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S33"]
        self.assertEqual(
            fixture["probe_words"],
            (*S33_WORDS, "洒溇", *S33_EXTERNAL_WORDS, S33_EXTERNAL_OCCUPANT[0]),
        )
        entries = {
            row["entry"]: tuple(row["pinyins"])
            for row in fixture["rows"]
            if row["kind"] == "entry"
        }
        self.assertEqual(entries, {
            "洒漏": ("sǎ", "lòu"),
            "撒漏": ("sǎ", "lòu"),
            "洒溇": ("sǎ", "lóu"),
            "缩手": ("suō", "shǒu"),
            "所售": ("suǒ", "shòu"),
            "所受": ("suǒ", "shòu"),
        })

    def test_s34_owns_the_exact_pending_batch_incident_fixture(self) -> None:
        self.assertEqual(S34_WORD, "开团")
        self.assertEqual(S34_PENDING_CODE, "khtt")
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S34"]
        self.assertEqual(fixture["probe_words"], (S34_WORD,))
        entry = next(
            row for row in fixture["rows"]
            if row["kind"] == "entry" and row["entry"] == S34_WORD
        )
        self.assertEqual(tuple(entry["pinyins"]), ("kāi", "tuán"))

    async def test_s29_seeds_the_exact_weighted_mkdr_chain(self) -> None:
        fixture = ZDIC_FIXTURES_BY_SCENARIO["S29"]
        self.assertEqual(fixture["probe_words"], ("火锅", "电脑"))
        self.assertEqual(S29_CODE, "mkdr")
        client = MagicMock()
        client.phrases_by_code = AsyncMock(side_effect=[
            [],
            [
                {"word": word, "code": S29_CODE, "type": "Phrase", "weight": weight}
                for word, weight in S29_CURRENT
            ],
        ])
        client.seed_phrase = AsyncMock(side_effect=[
            {"batchId": "seed-fire"},
            {"batchId": "seed-computer"},
        ])

        result = await ensure_s29_fixture(
            client=client,
            seed_identity={"platform_id": "9" * 32},
        )

        self.assertEqual(result["currentOrder"], ["火锅", "电脑"])
        self.assertEqual(
            [item.kwargs["weight"] for item in client.seed_phrase.await_args_list],
            [100, 101],
        )

    def test_bot_reference_fixture_uses_stable_full_vendored_database(self) -> None:
        class FakeBuildResult:
            def as_json_dict(self):
                return {
                    "commonness_word_count": 634829,
                    "corpus_word_count": 349045,
                }

        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = Path(temporary) / "runtime"
            with (
                patch(
                    "e2e.run.build_reference_database",
                    return_value=FakeBuildResult(),
                ) as build_mock,
                patch.dict("os.environ", {}, clear=False),
            ):
                result = build_bot_reference_fixture(runtime_dir)
                configured_path = result["databasePath"]

        self.assertEqual(
            build_mock.call_args.args[0],
            Path(__file__).parents[1] / "vendor" / "pinyin_reference",
        )
        self.assertEqual(build_mock.call_args.args[1], Path(configured_path))
        self.assertEqual(Path(configured_path), runtime_dir / "pinyin-reference.db")
        self.assertEqual(result["build"]["corpus_word_count"], 349045)

    async def test_s15_offline_scenario_contract(self) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S15")

        class FakeNextClient:
            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assert_token = admin_token
                if batch_id == "batch-numbered":
                    return {
                        "status": "Submitted",
                        "pullRequests": [
                            {
                                "action": "Create",
                                "word": "射覆",
                                "code": "eefju",
                            },
                        ],
                    }
                if batch_id == "batch-quoted":
                    return {
                        "status": "Submitted",
                        "pullRequests": [
                            {
                                "action": "Create",
                                "word": "亮面",
                                "code": "lxmmov",
                            },
                        ],
                    }
                raise AssertionError(batch_id)

        class FakeContext:
            fixture_facts = {
                "s15": {
                    "candidateCodes": ["eefj", "eefju", "eefjuv"],
                },
            }
            admin_token = "offline-admin-token"

            def __init__(self):
                self.events = []
                self.next_client = FakeNextClient()

            async def send(self, text: str) -> str:
                if text == "喵喵 射覆":
                    return "候选编码:\n1. eefj — 已有慑服\n2. eefju — 空位\n3. eefjuv — 空位"
                if text == "2 添加并提交":
                    self.events.append({
                        "sequence": 1,
                        "kind": "tool",
                        "name": "keytao_submit_batch",
                        "result": {"success": True, "batchId": "batch-numbered"},
                    })
                    return "✅ 射覆已加入草稿并提交审核"
                if text == "喵喵 亮面":
                    return "候选编码:\n1. lxmm — 已有占用\n2. lxmmov — 空位"
                if text == "添加并提交":
                    return (
                        "没有引用机器人给出的候选消息时，需要把词条和编码写完整，"
                        "请复制发送下面完整一行：\n"
                        "- 「添加 亮面 lxmmov 并提交」（亮面）"
                    )
                if text == "- 「添加 亮面 lxmmov 并提交」（亮面）":
                    self.events.append({
                        "sequence": 2,
                        "kind": "tool",
                        "name": "keytao_submit_batch",
                        "result": {"success": True, "batchId": "batch-quoted"},
                    })
                    return "✅ 亮面已加入草稿并提交审核"
                raise AssertionError(text)

            async def draft(self):
                return {"batchId": None, "contentVersion": 0, "items": []}

            def attempt_events(self):
                return list(self.events)

        result = await scenario.execute(FakeContext())
        self.assertEqual(result["facts"]["numberedCandidateCode"], "eefju")
        self.assertEqual(result["facts"]["suggestionSubcase"], "quoted-suggestion")
        self.assertEqual(
            result["facts"]["quotedSuggestion"],
            "- 「添加 亮面 lxmmov 并提交」（亮面）",
        )
        self.assertFalse(result["facts"]["additionalCorrectionRequired"])

    async def test_s15_direct_completion_satisfies_suggestion_subcase(self) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S15")

        class FakeNextClient:
            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assert_token = admin_token
                if batch_id == "batch-numbered":
                    return {
                        "status": "Submitted",
                        "pullRequests": [
                            {
                                "action": "Create",
                                "word": "射覆",
                                "code": "eefju",
                            },
                        ],
                    }
                if batch_id == "batch-direct":
                    return {
                        "status": "Submitted",
                        "pullRequests": [
                            {
                                "action": "Create",
                                "word": "亮面",
                                "code": "lxmm",
                                "needsManualReview": True,
                            },
                        ],
                    }
                raise AssertionError(batch_id)

        class FakeContext:
            fixture_facts = {
                "s15": {
                    "candidateCodes": ["eefj", "eefju", "eefjuv"],
                },
            }
            admin_token = "offline-admin-token"

            def __init__(self, *, require_confirmation: bool):
                self.events = []
                self.next_client = FakeNextClient()
                self.require_confirmation = require_confirmation

            def complete_submit(self) -> str:
                self.events.append({
                    "sequence": 2,
                    "kind": "tool",
                    "name": "keytao_submit_batch",
                    "result": {"success": True, "batchId": "batch-direct"},
                })
                return "✅ 搞定！「亮面」→ lxmm 已加入草稿并提交审核"

            async def send(self, text: str) -> str:
                if text == "喵喵 射覆":
                    return "候选编码:\n1. eefj — 已有慑服\n2. eefju — 空位\n3. eefjuv — 空位"
                if text == "2 添加并提交":
                    self.events.append({
                        "sequence": 1,
                        "kind": "tool",
                        "name": "keytao_submit_batch",
                        "result": {"success": True, "batchId": "batch-numbered"},
                    })
                    return "✅ 射覆已加入草稿并提交审核"
                if text == "喵喵 亮面":
                    return (
                        "候选编码:\n1. lxmm — ✅ 推荐（空位）\n2. lxmmo — 空位\n"
                        "是否以编码 lxmm 将「亮面」加入草稿？"
                    )
                if text == "添加并提交":
                    if self.require_confirmation:
                        return "提交前请核对服务端快照。\n回复「确认」执行，或「取消」。"
                    return self.complete_submit()
                if text == "确认" and self.require_confirmation:
                    return self.complete_submit()
                raise AssertionError(text)

            async def draft(self):
                return {"batchId": None, "contentVersion": 0, "items": []}

            def attempt_events(self):
                return list(self.events)

        for require_confirmation, expected_steps in ((False, 0), (True, 1)):
            with self.subTest(require_confirmation=require_confirmation):
                result = await scenario.execute(FakeContext(
                    require_confirmation=require_confirmation,
                ))
                self.assertEqual(
                    result["facts"]["suggestionSubcase"],
                    "direct-completion",
                )
                self.assertEqual(result["facts"]["directCompletionCode"], "lxmm")
                self.assertTrue(result["facts"]["directCompletionSealed"])
                self.assertEqual(
                    result["facts"]["additionalConfirmationSteps"],
                    expected_steps,
                )

    async def test_s16_offline_scenario_contract(self) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S16")

        class FakeNextClient:
            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assert_token = admin_token
                if batch_id != "batch-carrier":
                    raise AssertionError(batch_id)
                return {
                    "status": "Submitted",
                    "pullRequests": [
                        {
                            "action": "Create",
                            "word": "载流",
                            "code": "zhlq",
                        },
                        {
                            "action": "Delete",
                            "word": "座落在",
                            "code": "zlz",
                        },
                        {
                            "action": "Create",
                            "word": "座落在",
                            "code": "zlza",
                        },
                        {
                            "action": "Create",
                            "word": "载流子",
                            "code": "zlz",
                        },
                    ],
                }

        class FakeContext:
            fixture_facts = {
                "s16": {
                    "occupantWord": "座落在",
                    "occupiedCode": "zlz",
                    "shiftedCode": "zlza",
                },
            }
            admin_token = "offline-admin-token"

            def __init__(self, *, require_confirmation: bool, per_word_rendering: bool):
                self.events = []
                self.next_client = FakeNextClient()
                self.require_confirmation = require_confirmation
                self.per_word_rendering = per_word_rendering

            async def send(self, text: str) -> str:
                if text == "喵喵 加词 载流 载流子":
                    self.events.extend([
                        {
                            "sequence": 1,
                            "kind": "tool",
                            "name": "keytao_prepare_reviewed_add",
                            "result": {
                                "success": True,
                                "word": "载流",
                                "recommendedCode": "zhlq",
                                "needsManualReview": True,
                            },
                        },
                        {
                            "sequence": 2,
                            "kind": "tool",
                            "name": "keytao_prepare_reviewed_add",
                            "result": {
                                "success": True,
                                "word": "载流子",
                                "recommendedCode": "zlz",
                                "needsManualReview": False,
                            },
                        },
                        {
                            "sequence": 3,
                            "kind": "log",
                            "message": (
                                "Saved advertised reviewed batch candidate: "
                                "owner=fake items=2"
                            ),
                        },
                    ])
                    if self.per_word_rendering:
                        return (
                            "是否以编码 zhlq 将「载流」加入草稿？\n"
                            "是否以编码 zlz 将「载流子」加入草稿？\n"
                            "推荐：\n"
                            "- “「载流子」占 zlz、「座落在」顺延”（载流子、座落在）\n"
                            "不重排选 2（zlzu）。\n\n"
                            "回复「加入」、「都加」、「添加」只加入草稿；"
                            "回复「加入并提交」、「都加并提交」、「添加并提交」则加入后提交。\n"
                            "多个词的候选编号分别从 1 开始；选择时请带上词条，"
                            "例如「载流子 添加1」；多选请回复「载流子 添加2、4」。"
                        )
                    return (
                        "这些词是否一起加入草稿并提交？\n"
                        "- 「载流」→ zhlq\n"
                        "- 「载流子」→ zlz\n"
                        "推荐：\n"
                        "- “「载流子」占 zlz、「座落在」顺延”（载流子、座落在）\n"
                        "不重排选 2（zlzu）。\n\n"
                        "回复「加入」、「都加」、「添加」只加入草稿；"
                        "回复「加入并提交」、「都加并提交」、「添加并提交」则加入后提交。"
                    )
                if text == "加入并提交":
                    if self.require_confirmation:
                        return (
                            "提交前请核对服务端快照。\n"
                            "回复「确认」执行，或「取消」。"
                        )
                    self.events.append({
                        "sequence": 10,
                        "kind": "tool",
                        "name": "keytao_submit_batch",
                        "result": {"success": True, "batchId": "batch-carrier"},
                    })
                    return "✅ 载流、载流子已加入草稿并提交审核"
                if text == "确认" and self.require_confirmation:
                    self.events.append({
                        "sequence": 10,
                        "kind": "tool",
                        "name": "keytao_submit_batch",
                        "result": {"success": True, "batchId": "batch-carrier"},
                    })
                    return "✅ 载流、载流子已加入草稿并提交审核"
                raise AssertionError(text)

            async def draft(self):
                return {"batchId": None, "contentVersion": 0, "items": []}

            def attempt_events(self):
                return list(self.events)

        for per_word_rendering in (False, True):
            for require_confirmation, expected_steps in ((False, 0), (True, 1)):
                with self.subTest(
                    per_word_rendering=per_word_rendering,
                    require_confirmation=require_confirmation,
                ):
                    result = await scenario.execute(FakeContext(
                        require_confirmation=require_confirmation,
                        per_word_rendering=per_word_rendering,
                    ))
                    self.assertEqual(
                        result["facts"]["submittedWords"],
                        ["载流", "载流子"],
                    )
                    self.assertEqual(
                        result["facts"]["submittedCodes"],
                        ["zhlq", "zlz"],
                    )
                    self.assertFalse(result["facts"]["quoteRequired"])
                    self.assertEqual(
                        result["facts"]["additionalConfirmationSteps"],
                        expected_steps,
                    )

    async def test_s17_offline_scenario_contract(self) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S17")

        class FakeNextClient:
            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assert_token = admin_token
                if batch_id == "batch-semantic-pass":
                    return {
                        "status": "Approved",
                        "pullRequests": [{
                            "action": "Create",
                            "word": "产季",
                            "code": "jfjk",
                            "needsManualReview": False,
                        }],
                    }
                if batch_id == "batch-obscure-control":
                    return {
                        "status": "Submitted",
                        "pullRequests": [{
                            "action": "Create",
                            "word": "龘季",
                            "code": "dsjk",
                            "needsManualReview": True,
                        }],
                    }
                raise AssertionError(batch_id)

        class FakeContext:
            fixture_facts = {}
            admin_token = "offline-admin-token"

            def __init__(self):
                self.events = []
                self.next_client = FakeNextClient()

            async def send(self, text: str) -> str:
                if text == "喵喵 加词 产季":
                    return (
                        "审词：读音 chan ji；来源 本喵实体语境判断；"
                        "自动审核：该词可自动通过（语境读音与含义明确，"
                        "常用字组合且语义判断为常用或透明组合；"
                        "语料/词典证据：逐字 jieba 词频 产 6838、季 1619"
                        "（高频字阈值 1000），语义判断为常用或透明组合）\n"
                        "1. jfjk — ✅ 推荐（空位）\n"
                        "是否以编码 jfjk 将「产季」加入草稿？"
                    )
                if text == "添加并提交" and len(self.events) == 0:
                    self.events.append({
                        "sequence": 1,
                        "kind": "tool",
                        "name": "keytao_submit_batch",
                        "result": {
                            "success": True,
                            "batchId": "batch-semantic-pass",
                        },
                    })
                    return "✅ 产季已加入草稿并自动审核入库"
                if text == "喵喵 加词 龘季":
                    return (
                        "词库暂无收录「龘季」：\n"
                        "审词：读音 da ji；来源 暂无；\n"
                        "自动审核：非生僻条件不满足，需要管理员审核\n"
                        "候选编码:\n"
                        "1. dsjk — ✅ 推荐（空位）\n"
                        "• 「龘季」→ dsjk（推荐）\n"
                        "回复「加入」写入草稿，或回复「加入并提交」写入并提交。"
                    )
                if text == "添加并提交" and len(self.events) == 1:
                    self.events.append({
                        "sequence": 2,
                        "kind": "tool",
                        "name": "keytao_submit_batch",
                        "result": {
                            "success": True,
                            "batchId": "batch-obscure-control",
                        },
                    })
                    return "✅ 龘季已加入草稿并提交审核"
                raise AssertionError(text)

            async def draft(self):
                return {"batchId": None, "contentVersion": 0, "items": []}

            def attempt_events(self):
                return list(self.events)

        result = await scenario.execute(FakeContext())
        self.assertEqual(result["facts"]["semanticBatchStatus"], "Approved")
        self.assertFalse(result["facts"]["semanticNeedsManualReview"])
        self.assertEqual(result["facts"]["obscureBatchStatus"], "Submitted")
        self.assertTrue(result["facts"]["obscureNeedsManualReview"])

    async def test_s18_offline_incident_replay_uses_rendered_numbers(self) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S18")

        class FakeNextClient:
            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assert_token = admin_token
                if batch_id != "batch-s18":
                    raise AssertionError(batch_id)
                return {
                    "status": "Submitted",
                    "pullRequests": [
                        {
                            "action": "Create",
                            "word": "还车",
                            "code": "htjev",
                            "needsManualReview": False,
                        },
                        {
                            "action": "Create",
                            "word": "还车",
                            "code": "htwe",
                            "needsManualReview": True,
                        },
                    ],
                }

        class FakeContext:
            fixture_facts = {
                "s18": {"occupantWord": "换车", "occupiedCode": "htwe"},
            }
            admin_token = "offline-admin-token"

            def __init__(self):
                self.events = []
                self.items = []
                self.next_client = FakeNextClient()

            async def send(self, text: str) -> str:
                if text == "喵喵 还车":
                    return (
                        "词库暂无收录「还车」，先审读音和编码候选：\n\n"
                        "读音与来源:\n"
                        "1. huan che；来源 汉典（经编码服务） "
                        "https://www.zdic.net/hans/%E8%BF%98%E8%BD%A6\n"
                        "2. hái chē；来源 开放拼音数据（large_pinyin）；"
                        "汉典（离线数据集）\n"
                        "自动审核：该词可自动通过"
                        "（权威来源、编码和常用度证据一致）\n\n"
                        "候选编码（读音 1）:\n"
                        "1. htje — ✅ 推荐（空位）\n"
                        "2. htjev — 空位\n"
                        "3. htjevv — 空位\n"
                        "4. htwe — 已有「换车」\n"
                        "5. htwev — 空位\n"
                        "6. htwevv — 空位\n\n"
                        "候选编码（读音 2）:\n"
                        "7. hhje — ✅ 推荐（空位）\n"
                        "8. hhjev — 空位\n"
                        "9. hhjevv — 空位\n\n"
                        "是否以编码 htje 将「还车」加入草稿？可回复编号、编码，"
                        "或「都加」；可多选，如「添加2、4」。\n"
                        "若选的是已有词编码，回复“编号 重新编码”可挪开原词。\n"
                        "若所选编号显示“已有…”，直接回复该编号表示添加重码；"
                        "回复“编号 重新编码”或“原词 重新编码”则挪开原词。"
                    )
                if text == "添加2、99":
                    return "请选择 1-9 之间的编号；本次没有写入。"
                if text == "添加2、4":
                    self.events.append({
                        "sequence": 1,
                        "kind": "tool",
                        "name": "keytao_batch_add_to_draft",
                        "result": {
                            "success": False,
                            "requiresConfirmation": True,
                            "warningDigest": "a" * 64,
                            "warnings": [{
                                "warningType": "duplicate_code",
                                "item": {
                                    "action": "Create",
                                    "word": "还车",
                                    "code": "htwe",
                                },
                            }],
                        },
                    })
                    return "发现重码\n回复「确认」执行，或「取消」。"
                if text == "确认":
                    self.items = [
                        {
                            "action": "Create",
                            "word": "还车",
                            "code": "htjev",
                            "needsManualReview": False,
                        },
                        {
                            "action": "Create",
                            "word": "还车",
                            "code": "htwe",
                            "needsManualReview": True,
                        },
                    ]
                    return "✅ 已加入草稿"
                if text == "提交":
                    self.events.append({
                        "sequence": 2,
                        "kind": "tool",
                        "name": "keytao_submit_batch",
                        "result": {"success": True, "batchId": "batch-s18"},
                    })
                    return "✅ 草稿已提交审核"
                raise AssertionError(text)

            async def draft(self):
                return {
                    "batchId": "batch-s18" if self.items else None,
                    "contentVersion": 2 if self.items else 0,
                    "items": list(self.items),
                }

            def attempt_events(self):
                return list(self.events)

        result = await scenario.execute(FakeContext())
        self.assertEqual(
            result["messages"],
            ["喵喵 还车", "添加2、99", "添加2、4", "确认", "提交"],
        )
        self.assertEqual(result["facts"]["selectedIndexes"], [2, 4])
        self.assertEqual(result["facts"]["selectedCodes"], ["htjev", "htwe"])
        self.assertEqual(
            result["facts"]["candidateReadings"],
            ["huan che", "hái chē"],
        )
        self.assertEqual(result["facts"]["additionalConfirmationSteps"], 1)
        self.assertTrue(result["facts"]["duplicateWarningSealed"])

    async def test_s19_offline_incident_replay_chunks_and_writes_exact_remainder(
        self,
    ) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S19")

        class FakeContext:
            def __init__(self):
                self.events = []
                self.items = []
                self.batch_id = None

            async def send(self, text: str) -> str:
                if text.startswith("喵喵，请批量检查这些常用词"):
                    return (
                        "未收录词：" + "、".join(S19_ADVERTISED_WORDS)
                        + "。可以把这些词加入草稿。"
                    )
                if text == "火星词先不要，其他都加":
                    return "「火星词」不在刚才的候选中，请只从候选列表选择；本次未写入。"
                if text == "天选打工人先不要，其他可以加，沙县小吃也不要":
                    self.events.append({
                        "sequence": 1,
                        "kind": "message",
                        "direction": "reply",
                        "text": "正在处理「显眼包、嘴替、松弛感…」，已完成 8/9，预计还剩 1 轮",
                    })
                    return (
                        "已解析为以下 9 个词："
                        + "、".join(S19_ADVERTISED_WORDS[:-2])
                        + "\n回复「确认」执行，或「取消」。"
                    )
                if text == "确认":
                    self.items = [
                        {
                            "action": "Create",
                            "word": word,
                            "code": f"code{index}",
                        }
                        for index, word in enumerate(S19_ADVERTISED_WORDS[:-2])
                    ]
                    self.batch_id = "batch-s19"
                    self.events.append({
                        "sequence": 2,
                        "kind": "tool",
                        "name": "keytao_batch_add_to_draft",
                        "arguments": {"items": list(self.items)},
                        "result": {"success": True, "batchId": self.batch_id},
                    })
                    return "✅ 已加入草稿"
                raise AssertionError(text)

            async def draft(self):
                return {
                    "batchId": self.batch_id,
                    "contentVersion": 1 if self.items else 0,
                    "items": list(self.items),
                }

            def attempt_events(self):
                return list(self.events)

        result = await scenario.execute(FakeContext())
        self.assertEqual(result["facts"]["resolvedWords"], list(S19_ADVERTISED_WORDS[:-2]))
        self.assertEqual(result["facts"]["excludedWords"], ["天选打工人", "沙县小吃"])
        self.assertEqual(result["facts"]["confirmationSteps"], 1)
        self.assertEqual(result["facts"]["outOfSnapshotControl"], "ASK-without-write")
        self.assertEqual(result["facts"]["batchId"], "batch-s19")

    async def test_s20_offline_native_quote_writes_the_exact_advertised_batch(
        self,
    ) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S20")
        expected_pairs = (
            ("显眼包", "xybo"),
            ("嘴替", "zbtk"),
            ("松弛感", "swgv"),
        )

        class FakeContext:
            def __init__(self):
                self.events = []
                self.items = []
                self.batch_id = None
                self.last_reply_message_id = 501

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assert_to_me = to_me
                self.events.append({
                    "sequence": 1,
                    "kind": "log",
                    "message": "Saved advertised reviewed batch candidate: items=3",
                })
                return (
                    "建议批量加入：\n"
                    + "\n".join(
                        f'- 「{word}」 → {code}'
                        for word, code in expected_pairs
                    )
                    + "\n回复「加入」、「都加」、「添加」只加入草稿。"
                )

            async def send_group_reply(
                self,
                text: str,
                *,
                reply_message_id: int,
                to_me: bool,
            ) -> str:
                self.reply_request = (text, reply_message_id, to_me)
                self.items = [
                    {"action": "Create", "word": word, "code": code}
                    for word, code in expected_pairs
                ]
                self.batch_id = "batch-s20"
                self.events.append({
                    "sequence": 2,
                    "kind": "tool",
                    "name": "keytao_batch_add_to_draft",
                    "arguments": {"items": list(self.items)},
                    "result": {"success": True, "batchId": self.batch_id},
                })
                return "✅ 已加入草稿"

            async def draft(self):
                return {
                    "batchId": self.batch_id,
                    "contentVersion": 1 if self.items else 0,
                    "items": list(self.items),
                }

            def attempt_events(self):
                return list(self.events)

        context = FakeContext()
        result = await scenario.execute(context)
        self.assertTrue(context.assert_to_me)
        self.assertEqual(context.reply_request, ("都加", 501, True))
        self.assertEqual(result["facts"]["nativeQuoteMessageId"], 501)
        self.assertEqual(result["facts"]["advertisedPairs"], [list(pair) for pair in expected_pairs])
        self.assertEqual(result["facts"]["additionalConfirmationSteps"], 0)
        self.assertEqual(result["facts"]["batchId"], "batch-s20")

    async def test_s21_offline_replays_modifier_and_real_rendered_copy(self) -> None:
        from keytao_bot.harness.state import PendingToolConfirm
        from keytao_bot.plugins.chat_routing import (
            _format_live_ticket_precedence_message,
        )

        scenario = next(item for item in SCENARIOS if item.scenario_id == "S21")
        expected_pairs = tuple(
            (word, f"a{chr(ord('a') + index)}")
            for index, word in enumerate(S21_BATCH_WORDS)
        )

        class FakeContext:
            def __init__(self):
                self.events = []
                self.items = [
                    {"action": "Create", "word": "stale", "code": "stale"}
                ]
                self.batch_id = "batch-stale"
                self.sequence = 0
                self.next_client = self
                self.bot = self
                self.platform_id = "739497722"
                self.rendered_line = ""
                self.reset_calls = 0

            def add_write(self, pairs, batch_id):
                self.items = [
                    {"action": "Create", "word": word, "code": code}
                    for word, code in pairs
                ]
                self.batch_id = batch_id
                self.sequence += 1
                self.events.append({
                    "sequence": self.sequence,
                    "kind": "tool",
                    "name": "keytao_batch_add_to_draft",
                    "arguments": {"items": list(self.items)},
                    "result": {"success": True, "batchId": batch_id},
                })

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assert_to_me = to_me
                if text.startswith("喵喵 加词 "):
                    self.assertEqual(self.items, [])
                    return (
                        "建议批量加入：\n"
                        + "\n".join(
                            f'- 「{word}」 → {code}'
                            for word, code in expected_pairs
                        )
                        + "\n回复「加入」、「都加」、「添加」只加入草稿。"
                    )
                if text == "都加 跳过火星词":
                    return (
                        "「火星词」不在当前确认范围中；当前有效候选为「"
                        + "、".join(S21_BATCH_WORDS)
                        + "」；本次未写入。"
                    )
                if text == "都加 跳过嘴替":
                    self.add_write(expected_pairs[:-1], "batch-s21-modifier")
                    return "已按当前确认请求解析为以下 1 个词：" + "、".join(
                        S21_BATCH_WORDS[:-1]
                    )
                if text == "提交草稿":
                    state = PendingToolConfirm(
                        function_name="keytao_batch_add_to_draft",
                        args={
                            "items": [
                                {"action": "Create", "word": word, "code": code}
                                for word, code in expected_pairs
                            ],
                        },
                    )
                    guidance = _format_live_ticket_precedence_message(state)
                    self.rendered_line = "确认"
                    return guidance
                if text == "请阅读" + self.rendered_line:
                    return "这段引用不会授权写入。"
                if text == self.rendered_line:
                    self.add_write(expected_pairs, "batch-s21-rendered")
                    return "✅ 已加入草稿"
                raise AssertionError(text)

            async def draft(self):
                return {
                    "batchId": self.batch_id,
                    "contentVersion": 1 if self.items else 0,
                    "items": list(self.items),
                }

            async def clean_draft(self, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                deleted = len(self.items)
                self.items = []
                self.batch_id = None
                return {"success": True, "deleted": deleted}

            async def reset_conversation(self, *, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                self.reset_calls += 1

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError((left, right))

            def attempt_events(self):
                return list(self.events)

        context = FakeContext()
        result = await scenario.execute(context)

        self.assertEqual(context.reset_calls, 2)
        self.assertEqual(result["facts"]["excludedWord"], "嘴替")
        self.assertEqual(
            result["facts"]["resolvedWords"],
            list(S21_BATCH_WORDS[:-1]),
        )
        self.assertEqual(
            result["facts"]["renderedRemediationLine"],
            context.rendered_line,
        )
        self.assertEqual(context.rendered_line, "确认")
        self.assertEqual(result["facts"]["outOfTicketControl"], "ASK-without-write")
        self.assertEqual(
            result["facts"]["unrelatedQuoteControl"],
            "blocked-without-write",
        )
        self.assertEqual(result["facts"]["renderedBatchId"], "batch-s21-rendered")

    async def test_s22_offline_reestablishes_then_bare_assent_writes_displayed_set(
        self,
    ) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S22")
        expected_pairs = tuple(
            (word, f"a{chr(ord('a') + index)}")
            for index, word in enumerate(S22_BATCH_WORDS)
        )
        discovery_pairs = tuple(
            (word, code + "z")
            for word, code in expected_pairs
        )

        class FakeContext:
            admin_token = "offline-admin-token"

            def __init__(self):
                self.events = []
                self.completed_batch_id = None
                self.sequence = 0
                self.next_client = self
                self.bot = self
                self.platform_id = "739497722"
                self.last_reply_message_id = None
                self.reset_calls = 0
                self.reply_requests = []
                self.group_requests = []

            def record(
                self,
                *,
                kind,
                name="",
                arguments=None,
                result=None,
                message="",
            ):
                self.sequence += 1
                event = {"sequence": self.sequence, "kind": kind}
                if name:
                    event["name"] = name
                if arguments is not None:
                    event["arguments"] = arguments
                if result is not None:
                    event["result"] = result
                if message:
                    event["message"] = message
                self.events.append(event)

            async def clean_draft(self, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                return {"success": True, "deleted": 0}

            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assertEqual(batch_id, self.completed_batch_id)
                self.assertEqual(admin_token, self.admin_token)
                return {
                    "status": "Approved",
                    "pullRequests": [
                        {
                            "action": "Create",
                            "word": word,
                            "code": code,
                            "needsManualReview": False,
                        }
                        for word, code in expected_pairs
                    ],
                }

            async def reset_conversation(self, *, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                self.reset_calls += 1

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assertTrue(to_me)
                self.group_requests.append(text)
                if text.startswith("喵喵 加词 "):
                    self.last_reply_message_id = 501
                    return (
                        "首轮候选：\n"
                        + "\n".join(
                            f'- 「{word}」 → {code}'
                            for word, code in discovery_pairs
                        )
                        + "\n回复「加入并提交」则加入后提交。"
                    )
                if text == "加入并提交":
                    items = [
                        {"action": "Create", "word": word, "code": code}
                        for word, code in expected_pairs
                    ]
                    self.completed_batch_id = "batch-s22"
                    self.record(
                        kind="tool",
                        name="keytao_batch_add_to_draft",
                        arguments={"items": list(items)},
                        result={
                            "success": True,
                            "batchId": self.completed_batch_id,
                        },
                    )
                    self.record(
                        kind="tool",
                        name="keytao_submit_batch",
                        result={
                            "success": True,
                            "batchId": self.completed_batch_id,
                            "autoApproved": True,
                        },
                    )
                    return (
                        "✅ 批次已加入词库！\n"
                        + "\n".join(
                            f'- 「{word}」→ {code}'
                            for word, code in expected_pairs
                        )
                        + f"\n草稿地址：http://localhost:3100/batch/{self.completed_batch_id}"
                    )
                if text == "确认":
                    raise AssertionError("offline S22 should not need extra confirmation")
                raise AssertionError(text)

            async def send_group_reply(
                self,
                text: str,
                *,
                reply_message_id: int,
                to_me: bool,
            ) -> str:
                self.assertTrue(to_me)
                self.reply_requests.append((text, reply_message_id))
                if text.startswith("喵喵 请只重新复核以下 2 个词"):
                    self.assertEqual(reply_message_id, 501)
                    self.last_reply_message_id = 502
                    self.record(
                        kind="log",
                        message=(
                            "[advertised_reply_contract] "
                            "branch=establish_from_server_records items=2"
                        ),
                    )
                    return (
                        "2 个词已全部重新复核完毕。以下按候选列表格式逐项重列：\n\n"
                        + "\n\n".join(
                            f"{index}. 「{word}」\n"
                            f"   候选：1. {code} — 空位（推荐）｜"
                            f"2. {code}o — 空位"
                            for index, (word, code) in enumerate(
                                reversed(expected_pairs),
                                start=1,
                            )
                        )
                        + "\n\n可用的下一步：\n"
                        "- 「加入」、「都加」、「添加」→ 只加入草稿\n"
                        "- 「加入并提交」、「都加并提交」、「添加并提交」→ 加入后提交"
                    )
                raise AssertionError(text)

            async def draft(self):
                return {
                    "batchId": None,
                    "contentVersion": 0,
                    "items": [],
                }

            def attempt_events(self):
                return list(self.events)

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError((left, right))

            def assertTrue(self, value):
                if not value:
                    raise AssertionError(value)

        context = FakeContext()
        result = await scenario.execute(context)

        self.assertEqual(context.reset_calls, 2)
        self.assertEqual(
            context.reply_requests[0][1],
            result["facts"]["discoveryMessageId"],
        )
        self.assertEqual(len(context.reply_requests), 1)
        self.assertEqual(context.group_requests[-1], "加入并提交")
        self.assertTrue(result["facts"]["forcedStateLoss"])
        self.assertEqual(result["facts"]["confirmationSteps"], 0)
        self.assertEqual(result["facts"]["batchStatus"], "Approved")
        self.assertEqual(
            {tuple(pair) for pair in result["facts"]["advertisedPairs"]},
            set(expected_pairs),
        )
        self.assertEqual(
            {tuple(pair) for pair in result["facts"]["discoveryPairs"]},
            set(discovery_pairs),
        )
        self.assertEqual(result["facts"]["batchId"], "batch-s22")

    async def test_s23_offline_recovers_stale_quote_and_applies_same_assent(
        self,
    ) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S23")
        expected_pairs = tuple(
            (word, f"b{chr(ord('a') + index)}")
            for index, word in enumerate(S23_BATCH_WORDS)
        )
        persisted_pairs = (
            *expected_pairs[:6],
            expected_pairs[7],
            expected_pairs[6],
            *expected_pairs[8:],
        )

        class FakeContext:
            admin_token = "offline-admin-token"

            def __init__(self):
                self.events = []
                self.sequence = 0
                self.completed_batch_id = None
                self.next_client = self
                self.bot = self
                self.platform_id = "739497722"
                self.last_reply_message_id = None
                self.reset_calls = 0
                self.reply_requests = []

            def record(self, *, name, arguments=None, result=None):
                self.sequence += 1
                event = {
                    "sequence": self.sequence,
                    "kind": "tool",
                    "name": name,
                    "result": result or {},
                }
                if arguments is not None:
                    event["arguments"] = arguments
                self.events.append(event)

            async def clean_draft(self, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                return {"success": True, "deleted": 0}

            async def reset_conversation(self, *, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                self.reset_calls += 1

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assertTrue(to_me)
                if text.startswith("喵喵 加词 "):
                    self.last_reply_message_id = 601
                    return (
                        "陈旧候选：\n"
                        + "\n".join(
                            f'- 「{word}」 → {code}'
                            for word, code in expected_pairs
                        )
                        + "\n回复「加入并提交」则加入后提交。"
                    )
                if text == "确认":
                    raise AssertionError("offline S23 should not need extra confirmation")
                raise AssertionError(text)

            async def send_group_reply(
                self,
                text: str,
                *,
                reply_message_id: int,
                to_me: bool,
            ) -> str:
                self.assertTrue(to_me)
                self.reply_requests.append((text, reply_message_id))
                self.assertEqual((text, reply_message_id), ("加入并提交", 601))
                self.last_reply_message_id = 602
                items = [
                    {"action": "Create", "word": word, "code": code}
                    for word, code in persisted_pairs
                ]
                self.completed_batch_id = "batch-s23"
                self.record(
                    name="keytao_batch_add_to_draft",
                    arguments={"items": items},
                    result={"success": True, "batchId": self.completed_batch_id},
                )
                self.record(
                    name="keytao_submit_batch",
                    result={
                        "success": True,
                        "batchId": self.completed_batch_id,
                        "autoApproved": True,
                    },
                )
                return "✅ 已重新复核并按本轮指令加入并提交"

            async def draft(self):
                return {"batchId": None, "contentVersion": 0, "items": []}

            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assertEqual(batch_id, self.completed_batch_id)
                self.assertEqual(admin_token, self.admin_token)
                return {
                    "status": "Approved",
                    "pullRequests": [
                        {"action": "Create", "word": word, "code": code}
                        for word, code in persisted_pairs
                    ],
                }

            def attempt_events(self):
                return list(self.events)

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError((left, right))

            def assertTrue(self, value):
                if not value:
                    raise AssertionError(value)

        context = FakeContext()
        result = await scenario.execute(context)

        self.assertEqual(context.reset_calls, 2)
        self.assertEqual(context.reply_requests, [("加入并提交", 601)])
        self.assertEqual(result["facts"]["recoveryWrites"], 1)
        self.assertEqual(result["facts"]["confirmationSteps"], 0)
        self.assertEqual(result["facts"]["batchId"], "batch-s23")
        self.assertEqual(result["facts"]["batchStatus"], "Approved")
        self.assertEqual(
            result["facts"]["recoveredAppliedPairs"],
            [list(pair) for pair in expected_pairs],
        )

    async def test_s24_offline_quotes_single_candidate_and_submits_natural_assent(
        self,
    ) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S24")

        class FakeContext:
            admin_token = "offline-admin-token"

            def __init__(self):
                self.events = []
                self.sequence = 0
                self.completed_batch_id = None
                self.next_client = self
                self.bot = self
                self.platform_id = "3755240737"
                self.last_reply_message_id = None
                self.reset_calls = 0
                self.reply_requests = []

            def record(self, *, name, arguments=None, result=None):
                self.sequence += 1
                event = {
                    "sequence": self.sequence,
                    "kind": "tool",
                    "name": name,
                    "result": result or {},
                }
                if arguments is not None:
                    event["arguments"] = arguments
                self.events.append(event)

            async def clean_draft(self, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                return {"success": True, "deleted": 0}

            async def reset_conversation(self, *, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                self.reset_calls += 1

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assertTrue(to_me)
                if text == f"喵喵 {S24_WORD}":
                    self.last_reply_message_id = 701
                    return (
                        f"词库暂无收录「{S24_WORD}」，先审读音和编码候选：\n\n"
                        "候选编码:\n"
                        f"1. {S24_RECOMMENDED_CODE} — ✅ 推荐（空位）\n"
                        "2. htjev — 空位\n\n"
                        f"是否以编码 {S24_RECOMMENDED_CODE} 将「{S24_WORD}」加入草稿？\n"
                        "可回复编号或编码选择其他编码；可多选，如「添加1、2」。\n"
                        "回复「加入」只加入草稿；回复「加入并提交」则加入后提交。"
                    )
                if text == "确认":
                    raise AssertionError("offline S24 should not need extra confirmation")
                raise AssertionError(text)

            async def send_group_reply(
                self,
                text: str,
                *,
                reply_message_id: int,
                to_me: bool,
            ) -> str:
                self.assertTrue(to_me)
                self.reply_requests.append((text, reply_message_id))
                self.assertEqual((text, reply_message_id), (S24_NATURAL_ASSENT, 701))
                self.completed_batch_id = "batch-s24"
                arguments = {
                    "word": S24_WORD,
                    "code": S24_RECOMMENDED_CODE,
                    "confirmed": True,
                }
                self.record(
                    name="keytao_create_phrase",
                    arguments=arguments,
                    result={"success": True, "batchId": self.completed_batch_id},
                )
                self.record(
                    name="keytao_submit_batch",
                    result={
                        "success": True,
                        "batchId": self.completed_batch_id,
                        "autoApproved": True,
                    },
                )
                return "✅ 已加入草稿并提交审核"

            async def draft(self):
                return {"batchId": None, "contentVersion": 0, "items": []}

            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assertEqual(batch_id, self.completed_batch_id)
                self.assertEqual(admin_token, self.admin_token)
                return {
                    "status": "Approved",
                    "pullRequests": [{
                        "action": "Create",
                        "word": S24_WORD,
                        "code": S24_RECOMMENDED_CODE,
                    }],
                }

            def attempt_events(self):
                return list(self.events)

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError((left, right))

            def assertTrue(self, value):
                if not value:
                    raise AssertionError(value)

        context = FakeContext()
        result = await scenario.execute(context)

        self.assertEqual(context.reset_calls, 1)
        self.assertEqual(context.reply_requests, [(S24_NATURAL_ASSENT, 701)])
        self.assertEqual(result["facts"]["advertisedForms"], ["加入", "加入并提交"])
        self.assertEqual(result["facts"]["confirmationSteps"], 0)
        self.assertEqual(result["facts"]["batchId"], "batch-s24")
        self.assertEqual(result["facts"]["batchStatus"], "Approved")

    async def test_s25_offline_replays_natural_number_and_combined_submit(
        self,
    ) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S25")

        class FakeContext:
            admin_token = "offline-admin-token"

            def __init__(self):
                self.events = []
                self.sequence = 0
                self.items = []
                self.batch_id = None
                self.completed_batch_id = None
                self.next_client = self
                self.bot = self
                self.platform_id = "3755240737"
                self.reset_calls = 0
                self.cleanup_calls = 0

            def record(self, *, name, arguments=None, result=None):
                self.sequence += 1
                event = {
                    "sequence": self.sequence,
                    "kind": "tool",
                    "name": name,
                    "result": result or {},
                }
                if arguments is not None:
                    event["arguments"] = arguments
                self.events.append(event)

            async def clean_draft(self, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                deleted = len(self.items)
                self.items = []
                self.batch_id = None
                self.cleanup_calls += 1
                return {"success": True, "deleted": deleted}

            async def reset_conversation(self, *, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                self.reset_calls += 1

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assertTrue(to_me)
                if text == S25_NATURAL_ADD:
                    self.record(
                        name="keytao_create_phrase",
                        arguments={"word": S25_WORD, "code": S25_PREFIX_CODE},
                        result={
                            "success": False,
                            "requiresConfirmation": True,
                            "warningDigest": "a" * 64,
                            "message": "存在重码风险，请确认",
                        },
                    )
                    return f"检测到 {S25_PREFIX_CODE} 已有词条，确认后可继续重码写入。"
                if text == f"喵喵 {S25_WORD}":
                    return (
                        f"「{S25_WORD}」候选编码：\n"
                        f"1. {S25_SELECTED_CODE} — 空位 ✅\n"
                        "2. jlfo — 空位\n"
                        "3. jlfoo — 空位\n"
                        "4. jlfoou — 空位\n"
                        f"回复编号，或回复「用 {S25_SELECTED_CODE}」。"
                    )
                if text == "1":
                    self.batch_id = "batch-s25-number"
                    self.items = [{
                        "action": "Create",
                        "word": S25_WORD,
                        "code": S25_SELECTED_CODE,
                    }]
                    self.record(
                        name="keytao_create_phrase",
                        arguments={"word": S25_WORD, "code": S25_SELECTED_CODE},
                        result={"success": True, "batchId": self.batch_id},
                    )
                    return f"✅ 已将「{S25_WORD}」按 {S25_SELECTED_CODE} 加入草稿。"
                if text == S25_COMBINED_COMMAND:
                    self.completed_batch_id = "batch-s25-combined"
                    self.batch_id = self.completed_batch_id
                    self.items = [{
                        "action": "Create",
                        "word": S25_WORD,
                        "code": S25_SELECTED_CODE,
                    }]
                    arguments = {"word": S25_WORD, "code": S25_SELECTED_CODE}
                    self.record(
                        name="keytao_create_phrase",
                        arguments=arguments,
                        result={"success": True, "batchId": self.batch_id},
                    )
                    self.record(
                        name="keytao_submit_batch",
                        arguments={"batch_id": self.batch_id},
                        result={
                            "success": False,
                            "requiresConfirmation": True,
                            "batchId": self.batch_id,
                            "snapshotDigest": "b" * 64,
                        },
                    )
                    self.record(
                        name="keytao_submit_batch",
                        arguments={"batch_id": self.batch_id, "confirmed": True},
                        result={
                            "success": True,
                            "batchId": self.batch_id,
                            "autoApproved": True,
                        },
                    )
                    return (
                        "✅ 本轮已完成两步：\n"
                        f"- 已将「{S25_WORD}」 → {S25_SELECTED_CODE} "
                        "写入草稿。\n"
                        "- 已提交审核。"
                    )
                if text == "确认":
                    raise AssertionError("offline S25 should complete in the combined turn")
                raise AssertionError(text)

            async def draft(self):
                return {
                    "batchId": self.batch_id,
                    "contentVersion": 1 if self.items else 0,
                    "items": list(self.items),
                }

            async def get_admin_batch(self, *, batch_id: str, admin_token: str):
                self.assertEqual(batch_id, self.completed_batch_id)
                self.assertEqual(admin_token, self.admin_token)
                return {
                    "status": "Approved",
                    "pullRequests": [{
                        "action": "Create",
                        "word": S25_WORD,
                        "code": S25_SELECTED_CODE,
                    }],
                }

            def attempt_events(self):
                return list(self.events)

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError((left, right))

            def assertTrue(self, value):
                if not value:
                    raise AssertionError(value)

        context = FakeContext()
        result = await scenario.execute(context)

        self.assertEqual(context.cleanup_calls, 3)
        self.assertEqual(context.reset_calls, 3)
        self.assertEqual(
            result["messages"],
            [S25_NATURAL_ADD, f"喵喵 {S25_WORD}", "1", S25_COMBINED_COMMAND],
        )
        self.assertTrue(result["facts"]["naturalVerbReachedWriteGate"])
        self.assertTrue(result["facts"]["bareNumberWroteFromRecord"])
        self.assertEqual(result["facts"]["selectedIndex"], 1)
        self.assertEqual(result["facts"]["confirmationSteps"], 0)
        self.assertEqual(result["facts"]["batchId"], "batch-s25-combined")
        self.assertEqual(result["facts"]["batchStatus"], "Approved")

    async def test_s26_offline_replays_atomic_add_and_eviction(self) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S26")

        class FakeContext:
            def __init__(self):
                self.events = []
                self.items = []
                self.sequence = 0
                self.batch_id = ""
                self.platform_id = "s26-user"
                self.next_client = self
                self.bot = self
                self.fixture_facts = {"chixi_next_code": "wkxko"}
                self.cleanup_calls = 0
                self.reset_calls = 0

            async def clean_draft(self, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                self.items = []
                self.batch_id = ""
                self.cleanup_calls += 1
                return {"success": True}

            async def reset_conversation(self, *, platform_id: str):
                self.assertEqual(platform_id, self.platform_id)
                self.reset_calls += 1

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assertTrue(to_me)
                self.assertEqual(text, S26_COMMAND)
                self.batch_id = "batch-s26"
                self.items = [
                    {"action": "Delete", "word": S26_OCCUPANT, "code": S26_CODE},
                    {"action": "Create", "word": S26_WORD, "code": S26_CODE},
                    {
                        "action": "Create",
                        "word": S26_OCCUPANT,
                        "code": self.fixture_facts["chixi_next_code"],
                    },
                ]
                self.sequence += 1
                self.events.append({
                    "sequence": self.sequence,
                    "kind": "tool",
                    "name": "keytao_shift_phrase_code",
                    "arguments": {"word": S26_WORD, "target_code": S26_CODE},
                    "result": {
                        "success": True,
                        "batchId": self.batch_id,
                        "shiftPlan": {
                            "word": S26_WORD,
                            "targetCode": S26_CODE,
                            "shifted": [{
                                "word": S26_OCCUPANT,
                                "fromCode": S26_CODE,
                                "toCode": self.fixture_facts["chixi_next_code"],
                            }],
                        },
                    },
                })
                return (
                    "本轮已完成的写操作：\n"
                    f"- 已写入草稿：「{S26_WORD}」 → {S26_CODE}\n"
                    f"- 已顺延：「{S26_OCCUPANT}」 {S26_CODE} → "
                    f"{self.fixture_facts['chixi_next_code']}\n"
                    f"草稿/批次地址：http://localhost:3100/batch/{self.batch_id}"
                )

            async def draft(self):
                return {
                    "batchId": self.batch_id,
                    "contentVersion": 1,
                    "items": list(self.items),
                }

            def attempt_events(self):
                return list(self.events)

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError((left, right))

            def assertTrue(self, value):
                if not value:
                    raise AssertionError(value)

        context = FakeContext()
        result = await scenario.execute(context)

        self.assertEqual(context.cleanup_calls, 1)
        self.assertEqual(context.reset_calls, 1)
        self.assertEqual(result["messages"], [S26_COMMAND])
        self.assertEqual(result["facts"]["confirmationSteps"], 0)
        self.assertEqual(result["facts"]["batchId"], "batch-s26")
        self.assertEqual(result["facts"]["nextCode"], "wkxko")

    async def test_s37_offline_recovers_changed_selected_slot_without_looping(self) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S37")

        class FakeContext:
            def __init__(self):
                self.platform_id = "s37-user"
                self.next_client = self
                self.bot = self
                self.fixture_facts = {"s37": {"shiftedCode": "ppgv"}}
                self.items = []
                self.events = []
                self.clean_calls = 0
                self.reset_calls = 0
                self.injected = ""

            async def clean_draft(self, platform_id: str):
                self.assert_equal(platform_id, self.platform_id)
                self.items = []
                self.clean_calls += 1
                return {"success": True}

            async def reset_conversation(self, *, platform_id: str):
                self.assert_equal(platform_id, self.platform_id)
                self.reset_calls += 1

            async def phrases_by_word(self, word: str):
                self.assert_equal(word, S37_OCCUPANT)
                return [{
                    "word": S37_OCCUPANT,
                    "code": S37_TARGET_CODE,
                    "type": "Phrase",
                    "weight": 100,
                }]

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assert_true(to_me)
                if text == S37_COMMAND:
                    self.items = [
                        {"action": "Delete", "word": S37_OCCUPANT, "code": S37_TARGET_CODE},
                        {"action": "Create", "word": S37_WORD, "code": S37_TARGET_CODE},
                        {"action": "Create", "word": S37_OCCUPANT, "code": "ppgv"},
                    ]
                    return (
                        "本轮已完成的写操作：\n"
                        f"- 已写入草稿：「{S37_WORD}」 → {S37_TARGET_CODE}\n"
                        f"- 已顺延：「{S37_OCCUPANT}」 {S37_TARGET_CODE} → ppgv\n"
                        "草稿/批次地址：http://localhost:3100/batch/batch-s37"
                    )
                if text == f"喵喵 {S37_WORD}":
                    return (
                        f"「{S37_WORD}」候选编码：\n"
                        f"1. {S37_TARGET_CODE} — 已有「{S37_OCCUPANT}」\n"
                        "2. ppgv — 空位（推荐）"
                    )
                if text == S37_SELECTION:
                    return "已建立当前顺延计划；回复「确认」执行，或「取消」放弃。"
                raise AssertionError(text)

            def inject_bot_message(self, text: str) -> int:
                self.injected = text
                return 937

            async def send_group_reply(
                self,
                text: str,
                *,
                reply_message_id: int,
                to_me: bool,
            ) -> str:
                self.assert_equal((text, reply_message_id, to_me), (S37_SELECTION, 937, True))
                return (
                    "所选编码 zzzz 已不在当前候选中；"
                    "已刷新为当前候选，请按下面的列表重新选择。本次未写入。\n"
                    f"「{S37_WORD}」候选编码：\n"
                    f"1. {S37_TARGET_CODE} — 已有「{S37_OCCUPANT}」\n"
                    "2. ppgv — 空位（推荐）"
                )

            async def draft(self):
                return {
                    "batchId": "batch-s37" if self.items else None,
                    "items": list(self.items),
                }

            def attempt_events(self):
                return list(self.events)

            @staticmethod
            def assert_equal(actual, expected):
                if actual != expected:
                    raise AssertionError((actual, expected))

            @staticmethod
            def assert_true(value):
                if not value:
                    raise AssertionError(value)

        context = FakeContext()
        result = await scenario.execute(context)

        self.assertEqual(result["facts"]["verbatimEviction"], S37_COMMAND)
        self.assertEqual(result["facts"]["confirmationSteps"], 0)
        self.assertEqual(result["facts"]["shiftedCode"], "ppgv")
        self.assertTrue(result["facts"]["freshListRecovery"])
        self.assertFalse(result["facts"]["identicalRefusalRepeated"])
        self.assertEqual(context.clean_calls, 3)
        self.assertEqual(context.reset_calls, 4)
        self.assertIn("zzzz", context.injected)

    async def test_s39_offline_collapses_reading_selection_to_two_turns(self) -> None:
        scenario = next(item for item in SCENARIOS if item.scenario_id == "S39")

        class FakeContext:
            def __init__(self):
                self.platform_id = "s39-user"
                self.next_client = self
                self.bot = self
                self.fixture_facts = {"s39": {"shiftedCode": "jjqta"}}
                self.items = []
                self.events = []
                self.sequence = 0

            async def clean_draft(self, platform_id: str):
                self.assert_equal(platform_id, self.platform_id)
                self.items = []
                return {"success": True}

            async def reset_conversation(self, *, platform_id: str):
                self.assert_equal(platform_id, self.platform_id)

            async def phrases_by_word(self, word: str):
                self.assert_equal(word, S39_OCCUPANT)
                return [{
                    "word": S39_OCCUPANT,
                    "code": S39_TARGET_CODE,
                    "type": "Phrase",
                    "weight": 100,
                }]

            def _record(self, name: str, arguments: dict[str, Any]) -> None:
                self.sequence += 1
                self.events.append({
                    "sequence": self.sequence,
                    "kind": "tool",
                    "name": name,
                    "arguments": arguments,
                })

            async def send_group(self, text: str, *, to_me: bool) -> str:
                self.assert_true(to_me)
                if text == S39_COMMAND:
                    self._record("keytao_prepare_reviewed_add", {
                        "word": S39_WORD,
                        "requested_reading": "圈=quan",
                    })
                    return (
                        f"词库暂无收录「{S39_WORD}」：\n"
                        "审词：读音 chū quān；来源 用户当前指定 + 编码服务；\n"
                        "自动审核：指定读音与整词权威读音不同，需要管理员审核\n"
                        "候选编码:\n"
                        f"1. {S39_TARGET_CODE} — 已有「{S39_OCCUPANT}」\n"
                        "2. jjqta — 空位\n"
                        "3. jjqtai — 空位\n"
                        "回复编号或编码选择；回复“1 重新编码”挪开已有词。"
                    )
                if text in {S39_SELECTION, '重新编码 "除权" jjqt'}:
                    self._record("keytao_shift_phrase_code", {
                        "word": S39_WORD,
                        "target_code": S39_TARGET_CODE,
                        "confirmed_plan_digest": "a" * 64,
                    })
                    self.items = [
                        {"action": "Delete", "word": S39_OCCUPANT, "code": S39_TARGET_CODE},
                        {"action": "Create", "word": S39_WORD, "code": S39_TARGET_CODE,
                         "needsManualReview": True},
                        {"action": "Create", "word": S39_OCCUPANT, "code": "jjqta"},
                    ]
                    return (
                        f"已写入草稿：「{S39_WORD}」 → {S39_TARGET_CODE}；"
                        f"已顺延「{S39_OCCUPANT}」 {S39_TARGET_CODE} → jjqta。"
                    )
                if text == "加词 出圈 圈字读xing":
                    return (
                        "「出圈」的指定读音 xing 与编码服务返回的候选读音都不匹配。"
                        "可用读音：chū juàn、chū quān。"
                    )
                if text == "加词 出圈 jjqt 重新编码":
                    return (
                        "现有建议不能保留你要求的添加并腾位操作，"
                        "因此不提供缩窄后的命令；本次未写入。"
                    )
                raise AssertionError(text)

            async def draft(self):
                return {
                    "batchId": "batch-s39" if self.items else None,
                    "items": list(self.items),
                }

            def attempt_events(self):
                return list(self.events)

            @staticmethod
            def assert_equal(actual, expected):
                if actual != expected:
                    raise AssertionError((actual, expected))

            @staticmethod
            def assert_true(value):
                if not value:
                    raise AssertionError(value)

        result = await scenario.execute(FakeContext())

        self.assertEqual(result["facts"]["happyPathTurnCount"], 2)
        self.assertEqual(result["facts"]["selectionConfirmations"], 1)
        self.assertTrue(result["facts"]["unmatchedReadingListedAvailable"])
        self.assertTrue(result["facts"]["compoundSuggestionClosed"])
        self.assertTrue(result["facts"]["occupantPerspectiveResolved"])

    async def test_s27_offline_replays_binding_precheck_and_meta_answer(self) -> None:
        from keytao_bot.utils.pending_confirmation import (
            SYSTEM_REPLY_TEMPLATE_MARKERS,
            UNBOUND_BINDING_PRECHECK_NOTICE,
            single_word_candidate_footer,
        )

        scenario = next(item for item in SCENARIOS if item.scenario_id == "S27")

        class FakeContext:
            def __init__(self):
                self.platform_id = "9" * 32
                self.sender_name = "S27-bound"
                self.identity = {
                    "platform_id": self.platform_id,
                    "name": self.sender_name,
                }
                self.next_client = self
                self.bot = self
                self.base_url = "http://localhost:3100"
                self.events = []
                self.sequence = 0
                self.reset_ids = []

            async def find_user(self, platform_id: str):
                if platform_id == self.platform_id:
                    return {"id": "bound-user", "name": self.sender_name}
                return None

            async def reset_conversation(self, *, platform_id: str):
                self.reset_ids.append(platform_id)

            async def send_group(
                self,
                *,
                platform_id: str,
                sender_name: str,
                text: str,
                to_me: bool,
            ) -> str:
                self.assertTrue(to_me)
                if text == S27_WORD:
                    candidate = (
                        f"「{S27_WORD}」候选编码：\n"
                        "1. ldll — 空位 ✅\n\n"
                        f"是否以编码 ldll 将「{S27_WORD}」加入草稿？\n"
                        + single_word_candidate_footer(1)
                    )
                    return (
                        f"{UNBOUND_BINDING_PRECHECK_NOTICE.replace('keytao.vercel.app', 'keytao.rea.ink')}\n{candidate}"
                        if platform_id.startswith("8")
                        else candidate
                    )
                if text == S27_ASSENT:
                    self.assertTrue(platform_id.startswith("8"))
                    self.sequence += 1
                    self.events.append({
                        "sequence": self.sequence,
                        "kind": "tool",
                        "name": "keytao_create_phrase",
                        "result": {"success": False, "not_bound": True},
                    })
                    return (
                        "你还未绑定键道账号。\n"
                        "请发送 /bind 绑定码，详见 "
                        "https://keytao.rea.ink/profile"
                    )
                if text == S27_META_QUESTION:
                    self.assertTrue(platform_id.startswith("8"))
                    return (
                        "会的～简单说下我的实际流程：\n\n"
                        "• 查词、查编码、问规则这类只读操作：不需要绑定，谁都能用\n"
                        "• 加词、改词、提交草稿这类写操作：必须先绑定键道账号。"
                        "如果没绑定，工具层会直接拦截，词条根本进不了草稿，所以也就谈不上提交\n\n"
                        "之前你发「加入并提交」时就是这种情况：候选和确认都正常出来了，"
                        "但写入这一步被拦下了，所以我只能给你绑定指引。\n\n"
                        "绑定后同一句「加入并提交」就能正常执行了。"
                        "需要的话现在就可以 /bind 绑定～"
                    )
                raise AssertionError((platform_id, sender_name, text))

            async def draft(self):
                return {"batchId": None, "items": []}

            def attempt_events(self):
                return list(self.events)

            def assertTrue(self, value):
                if not value:
                    raise AssertionError(value)

        context = FakeContext()
        result = await scenario.execute(context)

        self.assertEqual(
            result["messages"],
            [S27_WORD, S27_ASSENT, S27_META_QUESTION, S27_WORD],
        )
        self.assertEqual(result["facts"]["bindingNoticeCount"], 1)
        self.assertEqual(result["facts"]["metaQuestionToolCalls"], 0)
        self.assertTrue(result["facts"]["boundControlNoticeAbsent"])
        self.assertFalse(
            any(
                marker in result["replies"][2]
                for marker in SYSTEM_REPLY_TEMPLATE_MARKERS
            )
        )
        self.assertEqual(
            context.reset_ids,
            ["8" + context.platform_id[1:], context.platform_id],
        )

    async def test_s23_unresolvable_quote_control_never_mints_a_write_ticket(
        self,
    ) -> None:
        from keytao_bot.harness.conversation import ConversationAddress
        from keytao_bot.harness.state import (
            MemoryConversationStateStore,
            PendingAdvertisedWordSets,
        )
        from keytao_bot.plugins import openai_chat
        from keytao_bot.plugins.chat_adapters import ReplyReferenceInfo
        from keytao_bot.utils.memory_store import ChatMemoryContext

        memory_context = ChatMemoryContext(
            platform="qq",
            user_id="s23-unresolved",
            space_type="group",
            space_id="865189947",
            speaker_name="Rea",
        )
        address = ConversationAddress.group("qq", "865189947", "s23-unresolved")
        quote = ReplyReferenceInfo(
            is_reply=True,
            is_to_bot=True,
            sender_id="bot-id",
            sender_name="喵喵",
            text=(
                "陈旧候选：\n"
                "- 「未知甲」 → forgeda\n"
                "- 「未知乙」 → forgedb\n"
                "回复「加入并提交」则加入后提交。"
            ),
        )
        ctx = openai_chat.TurnContext(
            bot=object(),
            event=object(),
            platform="qq",
            user_id="s23-unresolved",
            normalized_message_text="加入并提交",
            reply_reference=quote,
            memory_context=memory_context,
            conv_key=address,
            space_key=("qq", memory_context.space_scope_id),
            owner_label="Rea",
        )
        store = MemoryConversationStateStore()
        finish = AsyncMock()
        review = AsyncMock(
            return_value="以下词未通过完整审词，本次没有建立写入确认：未知甲、未知乙。"
        )
        with (
            patch.object(openai_chat, "conversation_state_store", store),
            patch.object(openai_chat, "get_history", return_value=[]),
            patch.object(openai_chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(openai_chat, "get_ai_response_core", review),
            patch.object(openai_chat, "remember_conversation", MagicMock()),
            patch.object(openai_chat, "_finish_ai_chat_matcher", finish),
        ):
            handled = await openai_chat._stage_apply_scoped_pending_intent(ctx)

        self.assertTrue(handled)
        self.assertEqual(
            review.await_args.kwargs["resolved_advertised_words"],
            ("未知甲", "未知乙"),
        )
        self.assertIsInstance(store.get(address), PendingAdvertisedWordSets)
        reply = finish.await_args.args[0]
        self.assertIn("- 「加词 未知甲 未知乙」（未知甲、未知乙）", reply)
        self.assertNotIn("显眼包", reply)
        self.assertNotIn("请重新", reply)

    async def test_s14_poison_injection_hooks_review_boundaries(self) -> None:
        from keytao_bot.utils import keytao_review as review_module

        review_module._clear_review_caches()
        controller = PronunciationPoisonController()
        controller.arm("S14")
        try:
            with (
                patch.object(
                    review_module,
                    "_request_bot_evidence_proxy",
                    AsyncMock(
                        return_value=review_module._BotEvidenceProxyResult(
                            "unavailable"
                        )
                    ),
                ),
                patch.object(
                    review_module,
                    "_search_web",
                    side_effect=controller.search_web,
                ),
                patch.object(
                    review_module,
                    "_fetch_text",
                    side_effect=controller.fetch_text,
                ),
            ):
                evidence = await review_module.collect_pronunciation_evidence("亮面")
        finally:
            controller.disarm()

        self.assertTrue(controller.injected)
        self.assertFalse(evidence["hasEvidence"])
        self.assertTrue(
            any(
                rejection.get("reason") == "queried_word_not_near_pinyin_label"
                and "光面" in unquote(str(rejection.get("url") or ""))
                for rejection in evidence["rejections"]
            )
        )

    async def test_multi_add_zdic_preflight_reuses_final_probe_retry(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        pinyin_by_char = {
            "王": "wáng",
            "中": "zhōng",
            "微": "wēi",
            "服": "fú",
            "务": "wù",
        }

        def probe_response(word: str, *, seeded: bool) -> dict[str, Any]:
            return {
                "input": word,
                "pronunciationSource": (
                    "pinyin-pro-context" if seeded else "zdic-unavailable"
                ),
                "standardPronunciationStatus": "absent" if seeded else "unavailable",
                "semanticPronunciationNeeded": not seeded,
                "chars": [
                    {
                        "char": char,
                        "pinyin": pinyin_by_char[char] if seeded else "",
                        "pinyins": [pinyin_by_char[char]] if seeded else [],
                        "pronunciationLookupStatus": (
                            "found" if seeded else "unavailable"
                        ),
                    }
                    for char in word
                ],
            }

        probe_words = ("王中王", "微服务")
        client.encode = AsyncMock(
            side_effect=[
                probe_response(word, seeded=attempt == 4)
                for attempt in range(1, 5)
                for word in probe_words
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ArtifactRecorder(Path(temp_dir) / "artifacts")
            with (
                recorder.scope("S10", 1),
                patch("e2e.run.asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
                patch("builtins.print"),
            ):
                result = await ensure_scenario_zdic_fixture(
                    client=client,
                    scenario_id="S10",
                    recorder=recorder,
                )
            artifact = json.loads(
                (recorder.artifact_dir / "S10-zdic-warmup-attempt-1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result["seededRealityMatches"])
        self.assertEqual(client.encode.await_count, 8)
        self.assertEqual(
            sleep_mock.await_args_list,
            [call(delay) for delay in S9_ZDIC_WARMUP_BACKOFF_SECONDS],
        )
        self.assertEqual(artifact["finalAssertionAttempt"], 4)
        self.assertEqual(artifact["finalAssertionResult"], "passed")
        self.assertEqual(
            set(artifact["attempts"][-1]["words"]),
            {"王中王", "微服务"},
        )

    async def test_s18_zdic_preflight_accepts_production_entry_shapes(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        authoritative_word = {
            "input": "还车",
            "pronunciationSource": "zdic-phrase",
            "standardPronunciationStatus": "found",
            "semanticPronunciationNeeded": False,
            "chars": [
                {
                    "char": "还",
                    "pinyin": "huán",
                    "pinyins": ["huán", "hái"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "车",
                    "pinyin": "chē",
                    "pinyins": ["chē"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        absent_occupant_word = {
            "input": "换车",
            "pronunciationSource": "pinyin-pro-context",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "chars": [
                {
                    "char": "换",
                    "pinyin": "huàn",
                    "pinyins": ["huàn"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "车",
                    "pinyin": "chē",
                    "pinyins": ["chē"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        client.encode = AsyncMock(
            side_effect=[
                response
                for _ in range(4)
                for response in (authoritative_word, absent_occupant_word)
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ArtifactRecorder(Path(temp_dir) / "artifacts")
            with (
                recorder.scope("S18", 1),
                patch("e2e.run.asyncio.sleep", new_callable=AsyncMock),
                patch("builtins.print"),
            ):
                result = await ensure_scenario_zdic_fixture(
                    client=client,
                    scenario_id="S18",
                    recorder=recorder,
                )
            artifact = json.loads(
                (recorder.artifact_dir / "S18-zdic-warmup-attempt-1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result["seededRealityMatches"])
        self.assertEqual(client.encode.await_count, 8)
        self.assertEqual(
            artifact["attempts"][-1]["words"]["还车"],
            {
                "pronunciationSource": "zdic-phrase",
                "standardPronunciationStatus": "found",
                "characterLookupStatuses": {"还": "found", "车": "found"},
                "seededRealityMatches": True,
            },
        )
        self.assertEqual(artifact["finalAssertionResult"], "passed")

    async def test_s27_zdic_preflight_accepts_seeded_local_entry_shape(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        seeded = {
            "input": S27_WORD,
            "pronunciationSource": "zdic-phrase",
            "standardPronunciationStatus": "found",
            "semanticPronunciationNeeded": False,
            "chars": [
                {
                    "char": "来",
                    "pinyin": "lái",
                    "pinyins": ["lái"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "都",
                    "pinyin": "dōu",
                    "pinyins": ["dōu", "dū"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "来",
                    "pinyin": "lái",
                    "pinyins": ["lái"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "了",
                    "pinyin": "le",
                    "pinyins": ["le", "liǎo"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        client.encode = AsyncMock(side_effect=[seeded] * 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ArtifactRecorder(Path(temp_dir) / "artifacts")
            with (
                recorder.scope("S27", 1),
                patch("e2e.run.asyncio.sleep", new_callable=AsyncMock),
                patch("builtins.print"),
            ):
                result = await ensure_scenario_zdic_fixture(
                    client=client,
                    scenario_id="S27",
                    recorder=recorder,
                )
            artifact = json.loads(
                (recorder.artifact_dir / "S27-zdic-warmup-attempt-1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result["seededRealityMatches"])
        self.assertEqual(client.encode.await_count, 4)
        self.assertEqual(
            artifact["attempts"][-1]["words"][S27_WORD],
            {
                "pronunciationSource": "zdic-phrase",
                "standardPronunciationStatus": "found",
                "characterLookupStatuses": {
                    "来": "found",
                    "都": "found",
                    "了": "found",
                },
                "seededRealityMatches": True,
            },
        )
        self.assertEqual(artifact["finalAssertionResult"], "passed")

    async def test_s14_zdic_probe_retries_transport_failure_then_passes(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        seeded = {
            "input": "亮面",
            "pronunciationSource": "pinyin-pro-context",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "chars": [
                {
                    "char": "亮",
                    "pinyin": "liàng",
                    "pinyins": ["liàng"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "面",
                    "pinyin": "miàn",
                    "pinyins": ["miàn"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        request = httpx.Request(
            "GET",
            "http://localhost:3100/api/phrases/encode?word=亮面",
        )
        client.encode = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("cold route compile", request=request),
                seeded,
                seeded,
                seeded,
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ArtifactRecorder(Path(temp_dir) / "artifacts")
            with (
                recorder.scope("S14", 1),
                patch("e2e.run.asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
                patch("builtins.print"),
            ):
                result = await ensure_scenario_zdic_fixture(
                    client=client,
                    scenario_id="S14",
                    recorder=recorder,
                )
            artifact = json.loads(
                (recorder.artifact_dir / "S14-zdic-warmup-attempt-1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result["seededRealityMatches"])
        self.assertEqual(client.encode.await_count, 4)
        self.assertEqual(
            sleep_mock.await_args_list,
            [call(delay) for delay in S9_ZDIC_WARMUP_BACKOFF_SECONDS],
        )
        self.assertEqual(
            artifact["attempts"][0]["transportError"],
            {"type": "ConnectTimeout", "message": "cold route compile"},
        )
        self.assertEqual(artifact["attempts"][1]["words"]["亮面"]["seededRealityMatches"], True)
        self.assertEqual(artifact["finalAssertionResult"], "passed")

    async def test_s14_zdic_probe_wraps_final_transport_failure(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        socket_stats = {
            "status": "captured",
            "targetPort": 3100,
            "tcpTimeWaitCount": 1200,
            "targetPortTimeWaitCount": 900,
        }
        request = httpx.Request(
            "GET",
            "http://localhost:3100/api/phrases/encode?word=亮面",
        )
        client.encode = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("connect cold", request=request),
                httpx.ReadTimeout("read cold", request=request),
                httpx.ConnectError("server restarting", request=request),
                httpx.ConnectTimeout("still cold", request=request),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ArtifactRecorder(Path(temp_dir) / "artifacts")
            with (
                recorder.scope("S14", 1),
                patch(
                    "e2e.run.collect_local_socket_stats",
                    return_value=socket_stats,
                ),
                patch("e2e.run.asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
                patch("builtins.print"),
            ):
                with self.assertRaisesRegex(
                    RigInfrastructureError,
                    "S14 ZDIC warm-up probe.*ConnectTimeout: still cold",
                ):
                    await ensure_scenario_zdic_fixture(
                        client=client,
                        scenario_id="S14",
                        recorder=recorder,
                    )
            artifact = json.loads(
                (recorder.artifact_dir / "S14-zdic-warmup-attempt-1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(client.encode.await_count, 4)
        self.assertEqual(
            sleep_mock.await_args_list,
            [call(delay) for delay in S9_ZDIC_WARMUP_BACKOFF_SECONDS],
        )
        self.assertEqual(
            [attempt["transportError"]["type"] for attempt in artifact["attempts"]],
            ["ConnectTimeout", "ReadTimeout", "ConnectError", "ConnectTimeout"],
        )
        self.assertEqual(artifact["localSocketStatsAtStart"], socket_stats)
        self.assertEqual(artifact["finalAssertionResult"], "failed")

    async def test_s9_zdic_preflight_only_asserts_seeded_reality_on_final_probe(
        self,
    ) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        unavailable = {
            "codes": ["wrong"],
            "pronunciationSource": "zdic-unavailable",
            "standardPronunciationStatus": "unavailable",
            "semanticPronunciationNeeded": True,
            "chars": [
                {"char": "射", "pronunciationLookupStatus": "unavailable"},
                {"char": "覆", "pronunciationLookupStatus": "unavailable"},
            ],
        }
        seeded = {
            "codes": ["eefj"],
            "altCodes": ["eefju", "eefjuv"],
            "pronunciationSource": "pinyin-pro-context",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "chars": [
                {
                    "char": "射",
                    "pinyin": "shè",
                    "pinyins": ["shè"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "覆",
                    "pinyin": "fù",
                    "pinyins": ["fù"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        occupant = {
            "word": "慑服",
            "code": "eefj",
            "type": "Phrase",
            "user": {"name": "existing-local-user"},
        }
        client.phrases_by_word = AsyncMock(return_value=[])
        client.phrases_by_code = AsyncMock(side_effect=[[occupant], [], []])
        client.encode = AsyncMock(side_effect=[unavailable, unavailable, unavailable, seeded])

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ArtifactRecorder(Path(temp_dir) / "artifacts")
            with (
                recorder.scope("S9", 1),
                patch("e2e.run.asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
                patch("builtins.print") as print_mock,
            ):
                result = await ensure_s9_fixture(
                    client=client,
                    seed_identity={"platform_id": "9" * 32},
                    admin_token="admin",
                    recorder=recorder,
                )

            artifact = json.loads(
                (recorder.artifact_dir / "S9-zdic-warmup-attempt-1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result["seededCharactersFound"])
        self.assertEqual(client.encode.await_count, 4)
        self.assertEqual(
            sleep_mock.await_args_list,
            [call(delay) for delay in S9_ZDIC_WARMUP_BACKOFF_SECONDS],
        )
        self.assertEqual(print_mock.call_count, 5)
        self.assertIn(
            "S9 zdic warm-up socket stats",
            print_mock.call_args_list[0].args[0],
        )
        self.assertEqual(artifact["finalAssertionAttempt"], 4)
        self.assertEqual(artifact["finalAssertionResult"], "passed")
        self.assertEqual(
            [attempt["pronunciationSource"] for attempt in artifact["attempts"]],
            ["zdic-unavailable"] * 3 + ["pinyin-pro-context"],
        )

    async def test_s16_dictionary_fixture_seeds_the_exact_incident_occupant(
        self,
    ) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        occupant = {
            "word": "座落在",
            "code": "zlz",
            "type": "Phrase",
            "weight": 100,
            "user": {"name": "keytao-e2e-llm-rig-run-seed"},
        }
        client.phrases_by_code = AsyncMock(side_effect=[[], [occupant], []])
        client.encode = AsyncMock(return_value={"codes": ["zlz", "zlza"]})
        client.clean_draft = AsyncMock(return_value={"success": True})
        client.seed_phrase = AsyncMock(return_value={"batchId": "fixture-batch"})

        result = await ensure_s16_fixture(
            client=client,
            seed_identity={"platform_id": "9" * 32},
        )

        client.clean_draft.assert_awaited_once_with("9" * 32)
        client.seed_phrase.assert_awaited_once_with(
            platform_id="9" * 32,
            word="座落在",
            code="zlz",
        )
        self.assertEqual(result["occupantWord"], "座落在")
        self.assertEqual(result["occupiedCode"], "zlz")
        self.assertEqual(result["shiftedCode"], "zlza")
        self.assertEqual(result["occupant"]["weight"], 100)

    async def test_s18_dictionary_fixture_seeds_the_exact_duplicate_occupant(
        self,
    ) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        occupant = {
            "word": "换车",
            "code": "htwe",
            "type": "Phrase",
            "weight": 100,
            "user": {"name": "keytao-e2e-llm-rig-run-seed"},
        }
        client.phrases_by_code = AsyncMock(side_effect=[[], [occupant]])
        client.clean_draft = AsyncMock(return_value={"success": True})
        client.seed_phrase = AsyncMock(return_value={"batchId": "fixture-batch"})

        result = await ensure_s18_fixture(
            client=client,
            seed_identity={"platform_id": "8" * 32},
        )

        client.clean_draft.assert_awaited_once_with("8" * 32)
        client.seed_phrase.assert_awaited_once_with(
            platform_id="8" * 32,
            word="换车",
            code="htwe",
        )
        self.assertEqual(result["occupantWord"], "换车")
        self.assertEqual(result["occupiedCode"], "htwe")
        self.assertEqual(result["occupant"]["weight"], 100)

    async def test_s25_dictionary_fixture_seeds_flykey_occupants_and_free_slot(
        self,
    ) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        wlf_occupant = {
            "word": "窝里反",
            "code": "wlf",
            "type": "Phrase",
            "weight": 100,
            "user": {"name": "keytao-e2e-llm-rig-run-seed"},
        }
        wlfo_occupant = {
            "word": "晚礼服",
            "code": "wlfo",
            "type": "Phrase",
            "weight": 100,
            "user": {"name": "keytao-e2e-llm-rig-run-seed"},
        }
        client.phrases_by_code = AsyncMock(side_effect=[
            [],
            [wlf_occupant],
            [],
            [wlfo_occupant],
            [],
        ])
        client.clean_draft = AsyncMock(return_value={"success": True})
        client.seed_phrase = AsyncMock(return_value={"batchId": "fixture-batch"})

        result = await ensure_s25_fixture(
            client=client,
            seed_identity={"platform_id": "7" * 32},
        )

        self.assertEqual(
            client.clean_draft.await_args_list,
            [call("7" * 32), call("7" * 32)],
        )
        self.assertEqual(
            client.seed_phrase.await_args_list,
            [
                call(platform_id="7" * 32, word="窝里反", code="wlf"),
                call(platform_id="7" * 32, word="晚礼服", code="wlfo"),
            ],
        )
        self.assertEqual(result["occupants"]["wlf"]["word"], "窝里反")
        self.assertEqual(result["occupants"]["wlfo"]["word"], "晚礼服")
        self.assertEqual(result["emptyCode"], "wlfoo")

    async def test_s41_dictionary_fixture_seeds_both_existing_codes(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        xjpoo = {
            "word": "畜产品",
            "code": "xjpoo",
            "type": "Phrase",
            "weight": 100,
            "user": {"name": "keytao-e2e-llm-rig-run-seed"},
        }
        jjpoo = {**xjpoo, "code": "jjpoo"}
        client.phrases_by_code = AsyncMock(side_effect=[[], [xjpoo], [], [jjpoo]])
        client.clean_draft = AsyncMock(return_value={"success": True})
        client.seed_phrase = AsyncMock(return_value={"batchId": "fixture-batch"})

        result = await ensure_s41_fixture(
            client=client,
            seed_identity={"platform_id": "6" * 32},
        )

        self.assertEqual(
            client.seed_phrase.await_args_list,
            [
                call(platform_id="6" * 32, word="畜产品", code="xjpoo"),
                call(platform_id="6" * 32, word="畜产品", code="jjpoo"),
            ],
        )
        self.assertEqual(result["word"], "畜产品")
        self.assertEqual(result["existingCodes"], ["xjpoo", "jjpoo"])

    def test_llm_endpoint_can_never_be_keytao_production(self) -> None:
        self.assertEqual(
            validate_llm_base("https://llm.example.com/v1"),
            "https://llm.example.com/v1/",
        )
        with self.assertRaises(SafetyViolation):
            validate_llm_base("https://keytao.vercel.app/v1")

    def test_binding_requires_reserved_metadata_and_non_qq_numeric_shape(self) -> None:
        user = {
            "name": "keytao-e2e-llm-rig-run-s1",
            "email": "keytao-e2e-llm-rig-run-s1@example.invalid",
            "roles": [{"value": "R:NORMAL"}, {"value": "R:BOT"}],
        }
        validate_test_binding(
            platform_id="9" * 32,
            expected_name=user["name"],
            expected_email=user["email"],
            user=user,
        )
        with self.assertRaises(SafetyViolation):
            validate_test_binding(
                platform_id="123456789",
                expected_name="ordinary-user",
                expected_email="person@example.com",
                user={
                    "name": "ordinary-user",
                    "email": "person@example.com",
                    "roles": [{"value": "R:NORMAL"}],
                },
            )

    def test_admin_requires_reserved_binding_and_real_admin_role(self) -> None:
        user = {
            "name": "keytao-e2e-llm-rig-run-admin",
            "email": "keytao-e2e-llm-rig-run-admin@example.invalid",
            "roles": [
                {"value": "R:NORMAL"},
                {"value": "R:BOT"},
                {"value": "R:MANAGER"},
            ],
        }
        validate_admin_identity(
            platform_id="9" * 32,
            expected_name=user["name"],
            expected_email=user["email"],
            user=user,
        )
        with self.assertRaises(SafetyViolation):
            validate_admin_identity(
                platform_id="9" * 32,
                expected_name=user["name"],
                expected_email=user["email"],
                user={**user, "roles": user["roles"][:2]},
            )
        with self.assertRaises(SafetyViolation):
            validate_admin_identity(
                platform_id="123456789",
                expected_name="admin",
                expected_email="admin@example.com",
                user={
                    "name": "admin",
                    "email": "admin@example.com",
                    "roles": [{"value": "R:ROOT"}],
                },
            )

    async def test_http_guard_rejects_production_before_transport(self) -> None:
        called = False

        async def transport_handler(_request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"unexpected": True})

        with patch.object(
            NetworkAllowlist,
            "_resolve_llm_ips",
            return_value=frozenset({"203.0.113.10"}),
        ):
            guard = NetworkAllowlist(llm_base_url="https://llm.example.com/v1")
        guard.install()
        try:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(transport_handler)
            ) as client:
                with self.assertRaises(SafetyViolation):
                    await client.get("https://keytao.vercel.app/api/phrases")
        finally:
            guard.restore()
        self.assertFalse(called)

    async def test_declared_dictionary_preflight_repairs_only_rig_owned_rows(
        self,
    ) -> None:
        rig_row = {
            "id": 31,
            "word": "射覆",
            "code": "eefju",
            "type": "Phrase",
            "user": {"name": "keytao-e2e-llm-rig-aborted-s15"},
        }
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        client.phrases_by_word = AsyncMock(side_effect=[[rig_row], [], [], []])
        client.clean_draft = AsyncMock(return_value={"success": True})
        client.add_draft_items = AsyncMock(
            return_value={"batchId": "cleanup", "contentVersion": 3}
        )
        client.submit_batch = AsyncMock(
            return_value={"submitted": {"batch": {"status": "Submitted"}}}
        )
        client.approve_admin_batch = AsyncMock(
            return_value={"batch": {"status": "Approved"}}
        )

        result = await repair_scenario_dictionary_fixture(
            client=client,
            scenario_id="S9",
            platform_id="9" * 32,
            admin_token="admin",
        )

        self.assertEqual(result["fixtureWords"], ["射覆", "慑服"])
        self.assertEqual(result["deletedFixtureRows"], 1)
        cleanup_items = client.add_draft_items.await_args.kwargs["items"]
        self.assertEqual(
            [(item["action"], item["word"], item["code"]) for item in cleanup_items],
            [("Delete", "射覆", "eefju")],
        )

        ordinary_row = {
            **rig_row,
            "id": 32,
            "user": {"name": "ordinary-user"},
        }
        refusing_client = LocalNextClient(
            base_url="http://localhost:3100",
            bot_token="test",
        )
        refusing_client.phrases_by_word = AsyncMock(return_value=[ordinary_row])
        refusing_client.clean_draft = AsyncMock()
        refusing_client.add_draft_items = AsyncMock()
        with self.assertRaisesRegex(
            RigInfrastructureError,
            "S9 requires 射覆 to be absent from the local dictionary",
        ):
            await repair_scenario_dictionary_fixture(
                client=refusing_client,
                scenario_id="S9",
                platform_id="9" * 32,
                admin_token="admin",
            )
        refusing_client.clean_draft.assert_not_awaited()
        refusing_client.add_draft_items.assert_not_awaited()

    async def test_s8_repair_ignores_prefix_rows_and_removes_rig_rows(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        rig_user = {"name": "keytao-e2e-llm-rig-old-s8"}
        shifted = {
            "id": 11,
            "word": "赤溪",
            "code": "wkxkv",
            "type": "Phrase",
            "weight": 100,
            "user": rig_user,
        }
        occupied = {
            "id": 12,
            "word": "吃席",
            "code": "wkxk",
            "type": "Phrase",
            "weight": 100,
            "user": rig_user,
        }
        restored = {
            "id": 13,
            "word": "赤溪",
            "code": "wkxk",
            "type": "Phrase",
            "weight": 100,
            "user": rig_user,
        }
        client.phrases_by_word = AsyncMock(side_effect=[[shifted], [restored]])
        client.phrases_by_code = AsyncMock(
            side_effect=[
                [occupied, shifted],
                [shifted],
                [restored],
                [],
            ]
        )
        client.clean_draft = AsyncMock(return_value={"success": True})
        client.add_draft_items = AsyncMock(
            return_value={"batchId": "cleanup", "contentVersion": 3}
        )
        client.submit_batch = AsyncMock(
            return_value={"submitted": {"batch": {"status": "Submitted"}}}
        )
        client.approve_admin_batch = AsyncMock(
            return_value={"batch": {"status": "Approved"}}
        )

        result = await client.restore_s8_fixture(
            platform_id="9" * 32,
            admin_token="admin",
            chixi_next_code="wkxkv",
        )

        self.assertTrue(result["verified"])
        self.assertEqual(result["deletedFixtureRows"], 2)
        cleanup_items = client.add_draft_items.await_args.kwargs["items"]
        self.assertEqual(
            [(item["action"], item["word"], item["code"]) for item in cleanup_items],
            [
                ("Delete", "吃席", "wkxk"),
                ("Delete", "赤溪", "wkxkv"),
                ("Create", "赤溪", "wkxk"),
            ],
        )

    async def test_s8_repair_refuses_non_rig_rows(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        ordinary_user = {"name": "ordinary-user"}
        shifted = {
            "id": 21,
            "word": "赤溪",
            "code": "wkxkv",
            "type": "Phrase",
            "weight": 100,
            "user": ordinary_user,
        }
        occupied = {
            "id": 22,
            "word": "吃席",
            "code": "wkxk",
            "type": "Phrase",
            "weight": 100,
            "user": ordinary_user,
        }
        client.phrases_by_word = AsyncMock(return_value=[shifted])
        client.phrases_by_code = AsyncMock(side_effect=[[occupied, shifted], [shifted]])
        client.clean_draft = AsyncMock()
        client.add_draft_items = AsyncMock()

        with self.assertRaises(RigInfrastructureError):
            await client.restore_s8_fixture(
                platform_id="9" * 32,
                admin_token="admin",
                chixi_next_code="wkxkv",
            )

        client.clean_draft.assert_not_awaited()
        client.add_draft_items.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
