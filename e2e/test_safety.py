"""Offline checks proving the E2E production guards fail before dispatch."""

from __future__ import annotations

import asyncio
import json
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
from .scenarios import SCENARIOS, S19_ADVERTISED_WORDS, S20_BATCH_WORDS
from .run import (
    S9_ZDIC_WARMUP_BACKOFF_SECONDS,
    abort_record_for_error,
    build_bot_reference_fixture,
    collect_local_socket_stats,
    ensure_s9_fixture,
    ensure_s16_fixture,
    ensure_s18_fixture,
    ensure_scenario_zdic_fixture,
    repair_scenario_dictionary_fixture,
)
from .runtime import E2EBotHarness, LocalNextClient, RigInfrastructureError
from .safety import (
    BLOCKED_EXTERNAL_DOMAINS,
    NetworkAllowlist,
    PronunciationPoisonController,
    SafetyViolation,
    validate_admin_identity,
    validate_keytao_base,
    validate_llm_base,
    validate_next_database_url,
    validate_test_binding,
)
from .zdic_seed import ZDIC_FIXTURES_BY_SCENARIO, seed_s9_zdic_cache, seed_zdic_cache


class SafetyRailTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertIn(
            "whole-word `corpus_frequency` and `common_characters_and_llm` routes",
            readme,
        )

    def test_scenario_pack_is_contiguous_through_s20(self) -> None:
        self.assertEqual(
            [scenario.scenario_id for scenario in SCENARIOS],
            [f"S{index}" for index in range(1, 21)],
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

    def test_bot_reference_fixture_uses_full_vendored_database(self) -> None:
        class FakeBuildResult:
            def as_json_dict(self):
                return {
                    "commonness_word_count": 634829,
                    "corpus_word_count": 349045,
                }

        with tempfile.TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary) / "artifacts"
            with (
                patch(
                    "e2e.run.build_reference_database",
                    return_value=FakeBuildResult(),
                ) as build_mock,
                patch.dict("os.environ", {}, clear=False),
            ):
                result = build_bot_reference_fixture(artifact_dir)
                configured_path = result["databasePath"]

        self.assertEqual(
            build_mock.call_args.args[0],
            Path(__file__).parents[1] / "vendor" / "pinyin_reference",
        )
        self.assertEqual(build_mock.call_args.args[1], Path(configured_path))
        self.assertTrue(configured_path.endswith("state/pinyin-reference.db"))
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
                        "请发送「添加 亮面 lxmmov 并提交」。"
                    )
                if text == "「添加 亮面 lxmmov 并提交」":
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
            "「添加 亮面 lxmmov 并提交」",
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
                        return "提交前请核对服务端快照。\n\n回复「确认」、「执行」继续。"
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
                            "action": "Create",
                            "word": "载流子",
                            "code": "zlzu",
                        },
                    ],
                }

        class FakeContext:
            fixture_facts = {}
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
                                "recommendedCode": "zlzu",
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
                            "是否以编码 zlzu 将「载流子」加入草稿？\n\n"
                            "回复「加入」、「都加」、「添加」只加入草稿；"
                            "回复「加入并提交」、「都加并提交」、「添加并提交」则加入后提交。\n"
                            "多个词的候选编号分别从 1 开始；选择时请带上词条，"
                            "例如「载流子 添加1」；多选请回复「载流子 添加2、4」。"
                        )
                    return (
                        "这些词是否一起加入草稿并提交？\n"
                        "- 「载流」→ zhlq\n"
                        "- 「载流子」→ zlzu\n\n"
                        "回复「加入」、「都加」、「添加」只加入草稿；"
                        "回复「加入并提交」、「都加并提交」、「添加并提交」则加入后提交。"
                    )
                if text == "加入并提交":
                    if self.require_confirmation:
                        return (
                            "提交前请核对服务端快照。\n\n"
                            "回复「确认」、「执行」继续。"
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
                        ["zhlq", "zlzu"],
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
                        "词库暂无收录「龘季」。\n"
                        "审词：读音 da ji；来源 本喵整词语境判断；"
                        "自动审核：该词需管理员审核（非生僻条件不满足）\n"
                        "1. dsjk — ✅ 推荐（空位）\n"
                        "是否以编码 dsjk 将「龘季」加入草稿？"
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
                    return "发现重码，回复「确认」、「执行」继续。"
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
                        + "\n确认后才会写入草稿。回复「确认」、「执行」继续。"
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

    async def test_s14_poison_injection_hooks_review_boundaries(self) -> None:
        from keytao_bot.utils import keytao_review as review_module

        review_module._clear_review_caches()
        controller = PronunciationPoisonController()
        controller.arm("S14")
        try:
            with (
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
        client.phrases_by_code = AsyncMock(side_effect=[[], [occupant]])
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
