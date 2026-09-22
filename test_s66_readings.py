"""Offline S66 transcript replays; all tools and model boundaries are fixtures."""

import copy
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import test_s63_fresh_selection as fresh
import test_s64_explicit_submit as explicit
from keytao_bot.harness.authorization_grammar import normalize_mutation_command_source
from keytao_bot.utils import keytao_review as review_module, observability
from keytao_bot.utils.reading_request import parse_parenthesised_readings


T1 = "@喵喵 一力降十会（yi2 li4 xiang3 shi2 hui），大力出奇迹（ da4 li4 chu1 qi2 ji1）"
T2 = "@喵喵 加词 @喵喵 一力降十会（yi2 li4 xiang2 shi2 hui4），大力出奇迹（ da4 li4 chu1 qi2 ji1）"
WORDS = ("一力降十会", "大力出奇迹")
READINGS = ("yī lì xiáng shí huì", "dà lì chū qí jì")
CODES = (("ylxh", "ylxhu", "ylxhui"), ("dlcj", "dlcju", "dlcjui"))


async def prepared_review(word, requested):
    index = WORDS.index(word)
    syllables = READINGS[index].split()
    alternate = ["yljh", "yljhu", "yljhui"] if index == 0 else []
    encode = {
        "success": True, "word": word, "codes": list(CODES[index]),
        "candidateCodes": [*CODES[index], *alternate],
        "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
        "phrasePinyins": syllables, "contextPhrasePinyins": syllables,
        "alternatePhrasePronunciationCodes": ([{
            "char": "降", "charIndex": 2, "pinyin": "jiàng", "codes": alternate,
        }] if alternate else []),
        "chars": [{"char": char, "pinyin": syllable,
                   "pinyins": [syllable, "jiàng"] if char == "降" else [syllable],
                   "pronunciationLookupStatus": "found"}
                  for char, syllable in zip(word, syllables)],
    }
    with patch.object(review_module, "collect_pronunciation_evidence_limited", AsyncMock(return_value={
        "success": True, "groups": [], "sources": [], "lookupComplete": True,
    })), patch.object(review_module, "fetch_keytao_encode", AsyncMock(return_value=encode)) as encoder, patch.object(
        review_module, "lookup_words", AsyncMock(return_value={}),
    ), patch.object(review_module, "lookup_codes", AsyncMock(return_value={})), patch.object(
        review_module, "AsyncOpenAI", Mock(side_effect=AssertionError("Review model forbidden")),
    ) as model:
        result = await review_module.prepare_reviewed_word(None, word, requested_reading=requested)
        encoder.assert_awaited_once()
        model.assert_not_called()
    return result


class TranscriptTests(unittest.IsolatedAsyncioTestCase):
    runtime = explicit.ExplicitSubmitTests.runtime
    turn = fresh.FreshSelectionTests.turn

    def setUp(self):
        fresh.FreshSelectionTests.setUp(self)
        self.key = fresh.harness.ConversationAddress.group("qq", "865189947", "s66-offline")
        self.memory = fresh.harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id=self.key.space_id,
        )
        self.items = []
        self.submitted = False
        self.version = 0
        self.batch_id = "s66-fixture"
        self.batch_url = "https://keytao.rea.ink/batch/s66-fixture"
        self.trace = []

    async def delivered_turn(self, message):
        token = observability.begin_turn_metrics("qq", "group")
        try:
            raw_response = response = await self.turn(message)
            response = fresh.chat._enforce_advertised_reply_contract(response, self.key)
            response = fresh.chat._prepare_user_facing_reply(response, self.memory)
            metrics = observability.current_turn_metrics()
            self.trace.append({"message": message, "flow": metrics.flow,
                               "modelCalls": metrics.model_calls, "rawReply": raw_response,
                               "reply": response})
            return response
        finally:
            observability.end_turn_metrics(token)

    async def tool(self, name, args, platform, user_id, **kwargs):
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_pending_items_by_words":
            result = {"success": True, "complete": True, "items": []}
        elif name == "keytao_lookup_by_word":
            result = {"success": True, "phrases": []}
        elif name == "keytao_prepare_reviewed_add":
            result = await prepared_review(args["word"], args.get("requested_reading", ""))
        elif name == "keytao_batch_add_to_draft":
            capabilities = kwargs["trusted_reviewed_items_by_key"]
            self.assertEqual(set(capabilities), {(word, codes[0]) for word, codes in zip(WORDS, CODES)})
            for word, codes, reading in zip(WORDS, CODES, READINGS):
                self.assertEqual(capabilities[(word, codes[0])]["pinyin"], reading)
            if not args.get("confirmed"):
                self.assertTrue(args["preview_only"])
                result = {"success": False, "requiresConfirmation": True,
                          "warningDigest": "a" * 64, "warnings": []}
            else:
                self.assertEqual(args["expected_content_version"], self.version)
                self.assertEqual(args["expected_warning_digest"], "a" * 64)
                self.assertFalse(self.writes)
                self.writes = [{"id": 6600 + index, **copy.deepcopy(row)}
                               for index, row in enumerate(args["items"])]
                self.items = copy.deepcopy(self.writes)
                self.version += 1
                result = {"success": True, "writtenItems": self.writes,
                          "successCount": 2, "failedCount": 0}
        elif name in {"keytao_list_draft_items", "keytao_get_batch_preview"}:
            result = {"success": True, "items": self.items, "count": len(self.items), "status": "Draft"}
        elif name == "keytao_submit_batch":
            if not args.get("confirmed"):
                result = {"success": False, "requiresConfirmation": True,
                          "snapshotDigest": "b" * 64, "warningDigest": "c" * 64,
                          "auditDigest": "d" * 64, "snapshotItems": self.items, "warnings": []}
            else:
                self.assertEqual(args["expected_content_version"], self.version)
                self.assertEqual(args["expected_server_snapshot_digest"], "b" * 64)
                self.assertEqual(args["expected_warning_digest"], "c" * 64)
                self.assertEqual(args["expected_audit_digest"], "d" * 64)
                self.assertFalse(self.submitted)
                self.submitted = True
                result = {"success": True, "status": "Submitted"}
        else:
            raise AssertionError("Unexpected tool: " + name)
        result.update(batchId=self.batch_id, batchUrl=self.batch_url, contentVersion=self.version)
        if kwargs.get("_capture_receipt", True):
            fresh.commands._capture_successful_draft_write_delivery(name, result, platform, user_id)
        return json.dumps(result, ensure_ascii=False)

    @contextmanager
    def executor_runtime(self):
        async def transport(name, args, context):
            self.assertEqual((context.platform, context.user_id), ("qq", self.key.actor_id))
            return json.loads(await self.tool(
                name, args, context.platform, context.user_id,
                trusted_reviewed_items_by_key=context.trusted_reviewed_items_by_key,
                _capture_receipt=False,
            ))

        with self.runtime(), patch.object(fresh.chat, "call_tool_function", explicit.REAL_TOOL_DISPATCH), patch.object(
            fresh.commands.tool_executor, "_invoke_effective_tool", side_effect=transport,
        ), patch.object(fresh.commands.memory_store, "record_tool_receipt"):
            yield

    async def test_exact_t1_and_t2_stay_deterministic(self):
        for message in (T1, T2):
            with self.subTest(message=message):
                self.setUp()
                with self.executor_runtime():
                    reply = await self.delivered_turn(message)
                    self.assertEqual(reply.count("审词："), 2, reply)
                    self.assertIn("加入并提交", reply)
                    self.assertTrue(fresh.chat._advertised_reply_matches_live_record(reply, self.store.get_record(self.key)))
                    self.assertIn("读音按标准", reply)
                    if message == T1:
                        self.assertIn("xiang3、hui", reply)
                    self.assertIn("yi2", reply)
                    self.assertEqual(self.trace[-1]["flow"], "word-discovery")
                    self.assertEqual(self.trace[-1]["modelCalls"], 0)
                    self.assertEqual(self.writes, [])
                    ticket = self.store.get(self.key)
                    self.assertEqual(ticket.args["_candidate_scopes"][0]["requestedReading"],
                                     "yi2 li4 xiang3 shi2 hui" if message == T1 else "yi2 li4 xiang2 shi2 hui4")
                    receipt = await self.delivered_turn("加入并提交")
                    self.assertIn("已提交审核", receipt)
                    self.assertTrue(self.submitted)
                    self.assertEqual([(row["word"], row["code"]) for row in self.writes],
                                     [(word, codes[0]) for word, codes in zip(WORDS, CODES)])
                    self.assertEqual(self.trace[-1]["modelCalls"], 0)
                args = [args for name, args in self.calls if name == "keytao_prepare_reviewed_add"]
                self.assertEqual([row["word"] for row in args], list(WORDS))
                self.assertEqual(args[0]["requested_reading"],
                                 "yi2 li4 xiang3 shi2 hui" if message == T1 else "yi2 li4 xiang2 shi2 hui4")
                self.assertEqual(args[1]["requested_reading"], "da4 li4 chu1 qi2 ji1")
                Path("/tmp/keytao-s66").mkdir(exist_ok=True)
                Path(f"/tmp/keytao-s66/{'t1' if message == T1 else 't2'}-replay.json").write_text(
                    json.dumps({"turns": self.trace, "calls": self.calls, "writes": self.writes,
                                "submitted": self.submitted}, ensure_ascii=False, indent=2) + "\n",
                )

    def test_three_mentions_are_removed_before_parsing(self):
        self.assertEqual(normalize_mutation_command_source("@喵喵 加词 @喵喵 小端、大端 @喵喵"), "加词  小端、大端")

    async def test_three_mentions_route_cleanly(self):
        with self.runtime():
            reply = await self.delivered_turn(T2 + " @喵喵")
        self.assertEqual(reply.count("审词："), 2)
        self.assertEqual(self.trace[-1]["modelCalls"], 0)

    async def test_marked_and_plain_readings_and_mixed_separators_route_without_model(self):
        for message in (
            "一力降十会( yī lì xiáng shí huì )和大力出奇迹（ dà lì chū qí jì ）",
            "添加 一力降十会（yi li xiang shi hui）；大力出奇迹( da li chu qi ji )",
            "加词 一力降十会（yī lì xiáng shí huì：ylxhu）、大力出奇迹（da4 li4 chu1 qi2 ji1）",
        ):
            self.setUp()
            with self.runtime():
                reply = await self.delivered_turn(message)
            self.assertEqual(reply.count("审词："), 2, reply)
            self.assertEqual(self.trace[-1]["modelCalls"], 0)

    async def test_item_cap_stops_before_tools(self):
        message = "，".join(f"词{char}（ci2 zi4）" for char in "甲乙丙丁戊己庚辛壬癸子")
        with self.runtime():
            reply = await self.delivered_turn(message)
        self.assertIn("最多查询 10 个词", reply)
        self.assertEqual(self.calls, [])

    async def test_foreign_actor_cannot_confirm_the_batch(self):
        with self.executor_runtime():
            await self.delivered_turn(T2)
            other = fresh.harness.ConversationAddress.group("qq", "865189947", "other-s66")
            await fresh.commands.handle_pending_message_core(
                "加入并提交", "qq", other.actor_id, other, allow_intent_model=False,
            )
            self.assertEqual(self.writes, [])
            self.assertFalse(self.submitted)

    async def test_conflicting_annotation_does_not_block_the_other_word(self):
        with self.runtime():
            reply = await self.delivered_turn(T1.replace("xiang3", "jiang4"))
        ticket = self.store.get(self.key)
        self.assertIn("音节差异", reply)
        self.assertIn("yljhui", reply)
        self.assertIn("加入并提交", reply)
        self.assertEqual([row["word"] for row in ticket.args["items"]], [WORDS[1]])

    async def test_t1_t2_t3_in_one_conversation(self):
        with self.executor_runtime():
            await self.delivered_turn(T1)
            reply = await self.delivered_turn(T2)
            self.assertEqual(reply.count("审词："), 2, self.trace)
            self.assertEqual(self.writes, [])
            receipt = await self.delivered_turn("加入并提交")
            self.assertIn("已提交审核", receipt)
            self.assertTrue(self.submitted)
            self.assertIn("用户标注读音 yi2 li4 xiang2 shi2 hui4", self.writes[0]["remark"])
            self.assertEqual([row["modelCalls"] for row in self.trace], [0, 0, 0])
        Path("/tmp/keytao-s66/t1-t2-t3-replay.json").write_text(
            json.dumps({"turns": self.trace, "calls": self.calls, "writes": self.writes},
                       ensure_ascii=False, indent=2) + "\n",
        )

    async def test_syllable_conflict_shows_both_chains_then_explicit_choice_resolves(self):
        with self.runtime():
            reply = await self.delivered_turn("一力降十会（yi1 li4 jiang4 shi2 hui4）")
            self.assertIn("音节差异", reply, self.trace)
            self.assertIn(READINGS[0], reply)
            self.assertIn("yī lì jiàng shí huì", reply)
            for code in (*CODES[0], "yljh", "yljhu", "yljhui"):
                self.assertIn(code, reply)
            self.assertEqual(reply.count("请明确"), 1)
            self.assertIsNone(self.store.get(self.key))
            selected = await fresh.commands._try_handle_explicit_reading_disambiguation(
                "一力降十会 读音 yi1 li4 jiang4 shi2 hui4", [], "qq", self.key.actor_id, self.key,
            )
            self.assertIn("yljh", selected)
            self.assertNotIn("请明确", selected)
            self.assertIsNotNone(self.store.get(self.key))
        self.assertEqual(self.writes, [])

    async def test_unavailable_syllable_keeps_known_chains_without_inventing_codes(self):
        with self.runtime():
            reply = await self.delivered_turn("一力降十会（yi1 li4 hong2 shi2 hui4）")
        self.assertIn("都不匹配", reply)
        self.assertIn("ylxhui", reply)
        self.assertIn("yljhui", reply)
        self.assertIsNone(self.store.get(self.key))

    async def test_explicit_parenthesised_code_is_bound_or_refused(self):
        for code in ("ylxhu", "zzzz"):
            self.setUp()
            with self.runtime():
                reply = await self.delivered_turn(f"添加 一力降十会（ yi2 li4 xiang3 shi2 hui ：{code} ）")
            if code == "ylxhu":
                self.assertEqual(self.store.get(self.key).args["items"][0]["code"], code)
                self.assertIn("读音按标准", reply)
            else:
                self.assertIn("不在该读音", reply)
                self.assertIn("ylxhui", reply)
                self.assertIn("yljhui", reply)
                self.assertEqual(reply.count("请明确"), 1)
                self.assertIsNone(self.store.get(self.key))


class ReadingGrammarTests(unittest.TestCase):
    def test_mentions_keep_other_users_and_lexical_substrings(self):
        for token in ("@喵喵", "喵喵", "键道", "@键道"):
            rows = parse_parenthesised_readings(
                f"{token} 加词 {token} 一力降十会(yi li xiang shi hui) {token}",
            )
            self.assertEqual(len(rows), 1)
        self.assertEqual(normalize_mutation_command_source("添加 @别人 喵喵叫( miao miao jiao )"),
                         "添加 @别人 喵喵叫( miao miao jiao )")
    def test_separators_tones_parentheses_and_inner_spaces(self):
        for prefix in ("", "加词 ", "添加 "):
            for separator in ("，", "、", "；", ",", ";", "和", " 和 "):
                for left, right in (("（", "）"), ("(", ")")):
                    for reading in ("yi2 li4 xiang3 shi2 hui", "yī lì xiáng shí huì", "yi li xiang shi hui"):
                        message = f"{prefix}一力降十会{left} {reading} {right}{separator}大力出奇迹{left}da4 li4 chu1 qi2 ji1{right}"
                        rows = parse_parenthesised_readings(message)
                        self.assertEqual(tuple(row.word for row in rows), WORDS, message)
                        self.assertEqual(rows[0].reading, reading)

    def test_malformed_or_extra_commands_do_not_gain_a_ticket(self):
        for message in (
            "不要加词 一力降十会(yi li xiang shi hui)",
            "他说 加词 一力降十会(yi li xiang shi hui)",
            "加词 一力降十会(yi li xiang shi hui)，删除小端",
            "一力降十会(yi li xiang shi hui)?", "一力降十会(yi li xiang shi hui",
            "一力降十会(yi li xiang shi hui), @别人 大力出奇迹(da li chu qi ji)",
            "一力降十会(yi li xiang shi hui)，一力降十会(yi li jiang shi hui)",
        ):
            self.assertFalse(parse_parenthesised_readings(message), message)

    def test_equivalent_tone_notation_does_not_claim_a_correction(self):
        result = review_module._requested_reading_context("yi1 li4 xiang2 shi2 hui4", READINGS[0])
        self.assertNotIn("readingCorrection", result)
        result = review_module._requested_reading_context("yi li xiang shi hui", READINGS[0])
        self.assertIn("readingCorrection", result)


if __name__ == "__main__":
    unittest.main()
