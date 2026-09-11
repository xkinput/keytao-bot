"""Offline S62 selection regressions; run in an isolated code snapshot."""

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from test_s54_selection import fixtures, reviewed_record
from keytao_bot.harness import authorization_grammar as grammar
from keytao_bot.plugins import chat_commands as commands, chat_routing as routing


def incident_record():
    """Adapt the existing sealed review fixture to the reported word and code."""
    record = reviewed_record()
    serialized = json.dumps(record.args, ensure_ascii=False)
    for old, new in (("大端", "加亮"), ("dsdtvo", "jslxa"),
                     ("dsdtv", "jslxo"), ("dsdt", "jslx")):
        serialized = serialized.replace(old, new)
    record.args = json.loads(serialized)
    return record


class S62SelectionTests(unittest.IsolatedAsyncioTestCase):
    async def execute_selection(self, message, record=None):
        store = fixtures.MemoryConversationStateStore()
        key = fixtures.ConversationAddress.private("qq", "s62-selection")
        store.set(key, record or incident_record())
        writes = []
        calls = []

        async def tool(name, args, platform, user_id, **_kwargs):
            self.assertEqual((platform, user_id), ("qq", "s62-selection"))
            if name in {"keytao_get_batch_preview", "keytao_list_draft_items"}:
                return json.dumps({
                    "success": True, "batchId": "s62-selection-batch",
                    "status": "Draft", "count": len(writes), "items": writes,
                })
            self.assertEqual(name, "keytao_batch_add_to_draft")
            calls.append(copy.deepcopy(args))
            if not args.get("confirmed"):
                self.assertTrue(args.get("preview_only"))
                return json.dumps({
                    "success": False, "requiresConfirmation": True,
                    "batchId": "s62-selection-batch", "contentVersion": 0,
                    "warningDigest": "a" * 64, "warnings": [],
                })
            self.assertEqual(args["batch_id"], "s62-selection-batch")
            self.assertEqual(args["expected_content_version"], 0)
            self.assertEqual(args["expected_warning_digest"], "a" * 64)
            self.assertFalse(writes, "A selection must not replay a confirmed write")
            writes.extend(copy.deepcopy(args["items"]))
            return json.dumps({
                "success": True, "batchId": "s62-selection-batch",
                "contentVersion": 1, "successCount": len(writes),
                "failedCount": 0, "pullRequestCount": len(writes),
                "writtenItems": [{"id": 6200 + index, **item}
                                 for index, item in enumerate(writes)],
            })

        with patch.object(fixtures.openai_chat_module, "conversation_state_store", store), patch.object(
            fixtures.openai_chat_module, "call_tool_function", side_effect=tool,
        ), patch.object(
            fixtures.openai_chat_module, "_classify_message_command_intent",
            AsyncMock(side_effect=AssertionError("S62 must not invoke a model")),
        ):
            response = await commands.handle_pending_message_core(
                message, "qq", "s62-selection", key, allow_intent_model=False,
            )
        return writes, calls, response, store.get(key)

    async def test_spaced_add_selector_writes_only_the_selected_live_word(self):
        record = incident_record()
        original = copy.deepcopy(record)
        writes, calls, response, pending = await self.execute_selection(
            "加亮 添加 jslxa", record,
        )
        self.assertEqual([(item["word"], item["code"]) for item in writes],
                         [("加亮", "jslxa")])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(pending)
        self.assertIn("小端", response)
        self.assertEqual(record, original)

    async def test_canonical_and_spaced_number_selectors_keep_the_same_write_scope(self):
        for message in ("加亮 jslxa", "加亮 3", "加亮 添加 3",
                        "加亮 jslxa，加入", "加亮 3，加入"):
            with self.subTest(message=message):
                writes, calls, response, pending = await self.execute_selection(message)
                self.assertEqual([(item["word"], item["code"]) for item in writes],
                                 [("加亮", "jslxa")])
                self.assertEqual(len(calls), 2)
                self.assertIsNone(pending)
                self.assertIn("加亮 → jslxa", response)
                self.assertIn("https://keytao.rea.ink/batch/s62-selection-batch", response)
                self.assertNotIn("已提交", response)

    async def test_comma_separated_choices_bind_each_words_own_inventory(self):
        for message in ("加亮 3，小端 xcdti", "加亮 jslxa、小端 2",
                        "加亮 添加 3，小端 xcdti", "加亮 jslxa、小端 添加 2"):
            with self.subTest(message=message):
                writes, _calls, _response, _pending = await self.execute_selection(message)
                self.assertEqual([(item["word"], item["code"]) for item in writes],
                                 [("加亮", "jslxa"), ("小端", "xcdti")])

    async def test_invalid_or_unbound_spaced_choices_never_write(self):
        for message in (
            "加亮 添加3", "加亮 添加jslxa", "加亮 添加 jslxa，加亮 2",
            "不要加亮 添加 jslxa", "他说加亮 添加 jslxa", "他说「加亮 添加 jslxa」",
            "「『加亮 添加 jslxa』」", "加亮 添加 jslxa？",
            "加亮 添加 jslxa，然后删除", "加亮 添加 jslxa；提交", "加亮 添加 3，",
            "未知 添加 1", "加亮 添加 evil", "加亮 添加 xcdti",
            "加亮 添加 0", "加亮 添加 4",
        ):
            with self.subTest(message=message):
                self.assertFalse(routing.message_authorizes_live_pending_mutation(
                    message, incident_record(),
                ))
                writes, calls, _response, _pending = await self.execute_selection(message)
                self.assertEqual(writes, [])
                self.assertEqual(calls, [])

    async def test_spaced_choices_still_require_the_live_review_seal(self):
        for mutation in ("payload", "scope", "missing", "server_warning"):
            with self.subTest(mutation=mutation):
                record = incident_record()
                scope = record.args["_candidate_scopes"][0]
                if mutation == "payload":
                    scope["reviewedState"]["word"] = "伪造"
                elif mutation == "scope":
                    scope["candidates"][2][0] = "forged"
                elif mutation == "missing":
                    scope.pop("reviewedState")
                else:
                    record.confirmation_source = "server_warning"
                for message in ("加亮 jslxa", "加亮 添加 jslxa"):
                    writes, calls, response, _pending = await self.execute_selection(message, record)
                    self.assertEqual(writes, [])
                    self.assertEqual(calls, [])
                    self.assertIn("未写入", response)

    async def test_occupied_selection_requires_exact_commonness_evidence(self):
        for verdict in ("front_more_common", "behind_more_common", "close", "uncertain", "missing", "wrong_word", "wrong_code", "unknown_occupant"):
            with self.subTest(verdict=verdict):
                record = incident_record()
                scope = record.args["_candidate_scopes"][0]
                occupants = {} if verdict == "unknown_occupant" else {"jslx": ["加量"]}
                assessments = [] if verdict == "missing" else [{
                    "newWord": "别词" if verdict == "wrong_word" else "加亮",
                    "occupantWord": "加量",
                    "occupantCode": "jslxo" if verdict == "wrong_code" else "jslx",
                    "verdict": "front_more_common" if verdict.startswith("wrong_") else verdict,
                }]
                scope["occupiedWords"] = scope["reviewedState"]["serverOccupiedWords"] = occupants
                scope["orderingAssessments"] = scope["reviewedState"]["serverOrderingAssessments"] = assessments
                original = copy.deepcopy(record)
                writes, calls, response, _ = await self.execute_selection("加亮 1", record)
                if verdict == "front_more_common":
                    self.assertEqual([(row["word"], row["code"]) for row in writes], [("加亮", "jslx")])
                    self.assertTrue(writes[0]["needsManualReview"])
                    self.assertIn("管理员审核", writes[0]["manualReviewReason"])
                    self.assertEqual(len(calls), 2)
                else:
                    self.assertEqual(writes, [])
                    self.assertEqual(calls, [])
                    self.assertIn("未加入", response)
                    self.assertIn("占用" if verdict == "unknown_occupant" else "常用度", response)
                    if verdict != "unknown_occupant":
                        self.assertIn("加量", response)
                self.assertEqual(record, original)

    async def test_protected_occupied_selection_does_not_block_other_selected_words(self):
        record = incident_record()
        scope = record.args["_candidate_scopes"][0]
        scope["occupiedWords"] = scope["reviewedState"]["serverOccupiedWords"] = {"jslx": ["加量"]}
        for message in ("加亮 1，小端 2", "小端 xcdti，加亮 jslx"):
            with self.subTest(message=message):
                writes, calls, response, pending = await self.execute_selection(message, record)
                self.assertEqual([(row["word"], row["code"]) for row in writes], [("小端", "xcdti")])
                self.assertEqual(len(calls), 2)
                self.assertIsNone(pending)
                self.assertIn("小端 → xcdti", response)
                self.assertIn("加量", response)
                self.assertIn("常用度", response)
                self.assertIn("未加入", response)
                self.assertIn("https://keytao.rea.ink/batch/s62-selection-batch", response)
                self.assertNotIn("已提交", response)

    def test_parser_rejects_quoted_or_joined_add_choices(self):
        self.assertEqual(grammar.parse_reviewed_multi_word_selection("加亮 添加 JSLXA"),
                         (("加亮", "jslxa"),))
        for message in ("「加亮 添加 jslxa」", "加亮 添加3", "加亮 添加jslxa"):
            with self.subTest(message=message):
                self.assertIsNone(grammar.parse_reviewed_multi_word_selection(message))


if __name__ == "__main__":
    unittest.main()
