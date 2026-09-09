"""S59 fixture-only route and delivery closure regressions."""
import asyncio
from contextlib import closing
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import test_state_machine as harness

chat = harness.openai_chat_module
commands = harness.chat_commands_module
MESSAGE = "请给这几个词的词频排序：耶博 伊莎贝拉 夜泊 一身本领"
WORDS = ("耶博", "伊莎贝拉", "夜泊", "一身本领")


class S59CommonnessRouteTests(unittest.TestCase):
    def setUp(self):
        from keytao_bot.utils import commonness_query as query
        from keytao_bot.utils import keytao_review as review
        self.query = query
        query.reset_commonness_delivery()
        self.addCleanup(query.reset_commonness_delivery)
        self.directory = tempfile.TemporaryDirectory(prefix="s59-route-")
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "reference.db"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, part_of_speech TEXT, dictionary_presence_count INTEGER)")
            connection.executemany("INSERT INTO word_commonness VALUES (?, ?, ?, ?)", [
                ("耶博", 100, "n", 2), ("伊莎贝拉", 1000, "nr", 3), ("夜泊", 400, "v", 3),
            ])
        patcher = patch.object(review, "reference_db_path", return_value=path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.owner = harness.ConversationAddress.private("qq", "s59-fixture")

    def test_closed_grammar_and_bounds(self):
        for message in (MESSAGE, "词频排序 耶博、伊莎贝拉，夜泊 一身本领",
                        "按常用度排序：耶博和夜泊", "哪个更常用：耶博 夜泊",
                        "常用度对比：耶博、夜泊", "耶博 和 夜泊 哪个更常用？"):
            with self.subTest(message=message):
                self.assertIsNotNone(self.query.parse_commonness_query(message))
        self.assertEqual(self.query.parse_commonness_query("常用度对比：和平 和 夜泊"), ("和平", "夜泊"))
        self.assertEqual(self.query.parse_commonness_query("词频排序：夜泊和耶博、伊莎贝拉"), ("夜泊", "耶博", "伊莎贝拉"))
        self.assertEqual(self.query.parse_commonness_query("词频排序：共和国 夜泊"), ("共和国", "夜泊"))
        self.assertEqual(self.query.parse_commonness_query("哪个更常用：添加 删除"), ("添加", "删除"))
        for message in ("词频排序 耶博", "词频排序 耶博 耶博", "他说词频排序：耶博 夜泊",
                        "不要词频排序：耶博 夜泊", "词频排序 耶博 夜泊；删除夜泊",
                        "词频排序 " + " ".join(chr(0x4e00 + i) for i in range(13))):
            self.assertIsNone(self.query.parse_commonness_query(message), message)
        self.assertEqual(len(self.query.parse_commonness_query("词频排序 " + " ".join(chr(0x4e00 + i) for i in range(12)))), 12)

    def test_incident_enters_zero_model_stage_before_classifiers(self):
        names = [stage.__name__ for stage in chat.STAGES]
        self.assertIn("_stage_handle_commonness_query", names)
        self.assertLess(names.index("_stage_handle_commonness_query"),
                        names.index("_stage_resolve_current_pending_scope"))

    async def run_incident(self, *, shared=True, missing=False, lookup_failure=False):
        calls = []
        rows = [
            {"word": "耶博", "code": "yebo", "type": "Phrase", "weight": 100},
            {"word": "夜泊", "code": "yebo" if shared else "abcd", "type": "Phrase", "weight": 101},
            {"word": "伊莎贝拉", "code": "yebl", "type": "Phrase", "weight": 100},
        ]
        if missing:
            rows = [row for row in rows if row["word"] != "夜泊"]

        async def dispatch(name, args, *actor):
            calls.append((name, args))
            self.assertEqual(actor, ("qq", self.owner.actor_id))
            if name == "keytao_lookup_by_words_batch":
                self.assertEqual(args, {"words": list(WORDS)})
                result = {"success": not lookup_failure, "results": [
                    {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                    for word in WORDS
                ]}
            elif name == "keytao_encode":
                self.assertEqual(set(args), {"word"})
                code = "yebo" if missing and args["word"] in ("夜泊", "耶博") else {
                    "耶博": "yebo", "夜泊": "abcd", "伊莎贝拉": "yebl", "一身本领": "none",
                }[args["word"]]
                result = {"success": True, "word": args["word"], "codes": [code, code + "a"]}
            else:
                raise AssertionError("S59 dispatched a write or model tool: " + name)
            return json.dumps(result)

        context = SimpleNamespace(conversation_address=self.owner, platform="qq")
        ctx = SimpleNamespace(normalized_message_text=MESSAGE, platform="qq", user_id=self.owner.actor_id,
                              conv_key=self.owner, memory_context=context, bot=object(), event=object(), QQMessageSegment=None)
        delivered = []

        async def finish(_bot, _event, _user, memory, response, _segment):
            delivered.append(chat._prepare_user_facing_reply(response, memory))

        token = chat._current_turn_message.set(MESSAGE)
        try:
            with (patch.object(commands, "call_tool_function", side_effect=dispatch),
                  patch.object(chat, "remember_conversation"),
                  patch.object(chat, "_finish_ai_chat_response", side_effect=finish),
                  patch.object(chat, "get_ai_response_core", side_effect=AssertionError("model is forbidden")),
                  patch.object(chat, "_classify_message_command_intent", side_effect=AssertionError("classifier is forbidden"))):
                self.assertTrue(await chat._stage_handle_commonness_query(ctx))
        finally:
            chat._current_turn_message.reset(token)
        self.assertEqual(len(delivered), 1)
        return delivered[0], calls

    def test_s59_fixture_scenario_table_affordance_and_delivery_closure(self):
        async def run():
            reply, calls = await self.run_incident()
            self.assertIn("名次 | 词 | 语料频次 | 词典收录 | 判定", reply)
            self.assertIn("1 | 伊莎贝拉 | 1000 | 3", reply)
            self.assertIn("2 | 夜泊 | 400 | 3", reply)
            self.assertIn("3 | 耶博 | 100 | 2", reply)
            self.assertIn("— | 一身本领 | — | — | 无数据", reply)
            suggestions = self.query.advertised_command_suggestions(reply)
            self.assertEqual(suggestions, ('把 夜泊 yebo 排到 耶博 yebo 前面',))
            parsed = self.query.parse_same_code_reorder(suggestions[0])
            self.assertEqual((parsed.first_word, parsed.second_word, parsed.first_code), ("夜泊", "耶博", "yebo"))
            self.assertTrue(self.query.matches_commonness_delivery(reply, self.owner, MESSAGE))
            self.assertFalse(self.query.matches_commonness_delivery(reply + "\n可发送「删除耶博」", self.owner, MESSAGE))
            self.assertFalse(self.query.matches_commonness_delivery(reply, harness.ConversationAddress.private("qq", "other"), MESSAGE))
            self.assertFalse(self.query.matches_commonness_delivery(reply, self.owner, "词频排序 甲 乙"))
            self.query.reset_commonness_delivery()
            self.assertFalse(self.query.matches_commonness_delivery(reply, self.owner, MESSAGE))
            self.assertTrue(all(name in {"keytao_lookup_by_words_batch", "keytao_encode"} for name, _ in calls))
            return reply
        self.last_reply = asyncio.run(run())

    def test_absent_word_uses_encoding_and_s50_parser(self):
        async def run():
            reply, calls = await self.run_incident(missing=True)
            suggestions = self.query.advertised_command_suggestions(reply)
            self.assertIn("把 夜泊 放在 耶博 前面", suggestions)
            parsed = self.query.parse_eviction_modified_add(suggestions[0])
            self.assertEqual((parsed.word, parsed.named_occupant), ("夜泊", "耶博"))
            self.assertIn(("keytao_encode", {"word": "夜泊"}), calls)
        asyncio.run(run())

    def test_no_shared_chain_or_failed_lookup_keeps_table_without_affordance(self):
        async def run():
            for options in ({"shared": False}, {"lookup_failure": True}):
                reply, _calls = await self.run_incident(**options)
                self.assertIn("夜泊 | 400 | 3", reply)
                self.assertFalse(self.query.advertised_command_suggestions(reply))
        asyncio.run(run())

    def test_ambiguous_same_code_types_and_divergent_suffixes_do_not_advertise(self):
        async def run():
            for ambiguous in (True, False):
                async def read(name, args):
                    if name == "keytao_lookup_by_words_batch":
                        return {"success": True, "results": [
                            {"word": word, "phrases": [
                                {"word": word, "code": "abcd" if ambiguous else code,
                                 "type": kind, "weight": 100}
                                for kind in (("Phrase", "CSS") if ambiguous else ("Phrase",))
                            ]}
                            for word, code in (("甲词", "abcxa"), ("乙词", "abcya"))
                        ]}
                    self.assertEqual(name, "keytao_encode")
                    return {"success": True, "word": args["word"], "codes": [
                        "abc", "abcxa" if args["word"] == "甲词" else "abcya",
                    ]}
                self.assertEqual(await self.query._placement_commands(
                    ("甲词", "乙词"), ["甲词", "乙词"], read,
                ), ())
        asyncio.run(run())

    def test_affordance_timeout_never_loses_local_ranking(self):
        async def run():
            with patch.object(self.query, "_placement_commands", side_effect=TimeoutError):
                reply, _calls = await self.run_incident()
            self.assertIn("夜泊 | 400 | 3", reply)
            self.assertIn("一身本领 | — | — | 无数据", reply)
            self.assertFalse(self.query.advertised_command_suggestions(reply))
        asyncio.run(run())

    def test_internal_fields_cannot_cross_actual_send_boundary(self):
        async def run():
            bot = SimpleNamespace(send=AsyncMock())
            for field in ("suggestedCommand", "blockReason", "boundTarget", "policyBlocked", "requiresTextFollowUp", "keytao_word_commonness"):
                raw = f"目前没有 {field}，请补充数据。"
                await chat._send_event_response(bot, object(), "s59", None, raw, emit_metrics=False)
                sent = bot.send.call_args.kwargs["message"]
                self.assertNotIn(field.lower(), sent.lower())
                self.assertTrue(sent)
                with self.assertRaises(ValueError):
                    chat._chat_render._assert_plain_user_facing_reply(raw)
        asyncio.run(run())

    def test_post_answer_augmentation_cannot_spend_a_third_model_call(self):
        from keytao_bot.utils.observability import begin_turn_metrics, current_turn_metrics, end_turn_metrics

        async def run():
            token = begin_turn_metrics("qq", "private")
            try:
                current_turn_metrics().model_calls = 2
                ctx = SimpleNamespace(response="现有回答", generic_command_intent=SimpleNamespace(intent="none"),
                                      normalized_message_text="日常问题")
                with patch.object(chat, "_get_simple_word_query_words", side_effect=AssertionError("third model call")):
                    self.assertFalse(await chat._stage_augment_word_query(ctx))
                self.assertEqual(ctx.response, "现有回答")
            finally:
                end_turn_metrics(token)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
