"""Exercise explicit same-code requests through the real plan and ticket seam."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness

commands = harness.chat_commands_module
chat = harness.openai_chat_module
draft_tools = harness._draft_tools


class SameCodeReorderTests(unittest.TestCase):
    def test_non_same_code_relative_request_remains_available_to_existing_route(self):
        from keytao_bot.harness.authorization_grammar import parse_eviction_modified_add
        from keytao_bot.utils.same_code_reorder import try_handle_same_code_reorder
        message = "把小像放到销项前面"
        existing_grammar = parse_eviction_modified_add(message)
        self.assertIsNotNone(existing_grammar)
        self.assertEqual(
            (existing_grammar.word, existing_grammar.named_occupant), ("小像", "销项"),
        )

        async def run():
            for first_rows in ([], [{"word": "小像", "code": "abcdo", "type": "Phrase", "weight": 100}]):
                async def call(name, args, *_args, **_kwargs):
                    if name == "keytao_lookup_by_words_batch":
                        return json.dumps({"success": True, "results": [
                            {"word": "小像", "phrases": first_rows},
                            {"word": "销项", "phrases": [{"word": "销项", "code": "abcd", "type": "Phrase", "weight": 100}]},
                        ]})
                    if name == "keytao_list_draft_items":
                        return json.dumps({"success": True, "batchId": "", "contentVersion": 0, "items": []})
                    raise AssertionError("non-same-code request entered the same-code planner")
                with patch.object(commands, "call_tool_function", side_effect=call):
                    self.assertIsNone(await try_handle_same_code_reorder(message, "qq", "s57-relative"))
                    explicit = await try_handle_same_code_reorder(
                        "把「小像 abcd」放到「销项 abcd」前面", "qq", "s57-relative",
                    )
                    self.assertIn("未能锁定", explicit)
        asyncio.run(run())

    def test_applied_single_weight_receipt_lists_both_verified_changes(self):
        # Minimized successful tool result captured from S57 target run 3.
        result = {
            "success": True, "batchId": "s57-receipt", "successCount": 2,
            "receipts": [{"step": "dictionary", "status": "applied", "changes": [], "itemCount": 2}],
            "shiftPlan": {
                "word": "嘢", "targetCode": "yeoiav", "scope": "same_code",
                "currentState": [
                    {"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 11, "source": "live"},
                    {"word": "咽", "code": "yeoiav", "type": "Single", "weight": 10, "source": "live"},
                ],
                "proposedState": [
                    {"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 10},
                    {"word": "咽", "code": "yeoiav", "type": "Single", "weight": 11},
                ],
                "shifted": [], "draftUpdates": [],
            },
            "draft_snapshot": {
                "success": True, "batchId": "s57-receipt", "contentVersion": 1,
                "items": [
                    {"action": "Change", "word": "嘢", "oldWord": "嘢", "code": "yeoiav", "type": "Single", "weight": 10},
                    {"action": "Change", "word": "咽", "oldWord": "咽", "code": "yeoiav", "type": "Single", "weight": 11},
                ],
            },
        }
        reply = commands._format_ranked_shift_success(result)
        self.assertIn("嘢 yeoiav 权重 11→10", reply)
        self.assertIn("咽 yeoiav 权重 10→11", reply)
        self.assertNotIn("嘢 → yeoiav", reply)
        # A proposed state alone cannot prove that an individual row changed.
        result["draft_snapshot"]["items"].pop()
        self.assertNotIn("咽 yeoiav 权重 10→11", commands._format_ranked_shift_success(result))

    def test_closed_parser_accepts_requested_forms_without_claiming_reported_text(self):
        from keytao_bot.utils.same_code_reorder import parse_same_code_reorder
        forms = (
            '提高"嘢    yeoiav"的优先级，使之位于"咽    yeoiav"之前',
            '提高＂嘢 yeoiav＂的优先级，使之位于＂咽 yeoiav＂之前',
            "提高「嘢 yeoiav」的优先级，使之位于「咽 yeoiav」之前",
            "把 嘢 排到 咽 前面", "把财宝排到财报之前",
            "嘢 和 咽 调换顺序", "嘢和咽对换", "嘢优先于咽",
            "对换财宝和财报的优先级",
        )
        for text in forms:
            with self.subTest(text=text):
                self.assertIsNotNone(parse_same_code_reorder(text))
        for text in (
            "不要把嘢排到咽前面", "他说把嘢排到咽前面", "把嘢排到咽前面？",
            "把嘢排到咽前面 删除咽", "嘢和嘢对换", "强制调序", "四字词语",
            '提高"嘢 yeoiav」的优先级，使之位于"咽 yeoiav"之前',
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_same_code_reorder(text))

    def test_full_chain_preserves_unmentioned_entries_and_typed_base(self):
        from keytao_bot.utils.same_code_reorder import try_handle_same_code_reorder
        async def run():
            for phrase_type, base, words in (("Single", 10, ("甲", "乙", "丙")), ("Phrase", 100, ("甲词", "乙词", "丙词"))):
                for tied in (False, True):
                    with self.subTest(phrase_type=phrase_type, tied=tied):
                        rows = [
                            {"word": word, "code": "abcd", "type": phrase_type, "weight": base if tied else base + index}
                            for index, word in enumerate(words)
                        ]
                        async def call(name, args, *_args, **_kwargs):
                            if name == "keytao_lookup_by_words_batch":
                                result = {"success": True, "results": [
                                    {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                                    for word in args["words"]
                                ]}
                            elif name == "keytao_lookup_by_code":
                                result = {"success": True, "phrases": rows}
                            elif name == "keytao_list_draft_items":
                                result = {"success": True, "batchId": "", "contentVersion": 0, "items": []}
                            else:
                                raise AssertionError(name)
                            return json.dumps(result)
                        shift = AsyncMock(return_value="preview")
                        with (
                            patch.object(commands, "call_tool_function", side_effect=call),
                            patch.object(commands, "_execute_shift_to_code", shift),
                        ):
                            await try_handle_same_code_reorder(f"把{words[2]}排到{words[0]}前面", "qq", "s57-chain")
                        args = shift.call_args.kwargs
                        self.assertEqual(set(args["ordered_words"]), set(words))
                        self.assertLess(args["ordered_words"].index(words[2]), args["ordered_words"].index(words[0]))
                        self.assertEqual(args["expected_codes"], ["abcd"] * 3)
                        self.assertEqual(args["expected_weights"], None if tied else [base, base + 1, base + 2])
                        self.assertEqual(sum(line.startswith("常用度提示：") for line in args["evidence_lines"]), 1)
        asyncio.run(run())

    def test_explicit_same_code_scope_keeps_other_codes_untouched(self):
        async def run():
            rows = [
                {"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 11},
                {"word": "嘢", "code": "yeoia", "type": "Single", "weight": 10},
                {"word": "咽", "code": "yeoiav", "type": "Single", "weight": 10},
                {"word": "其他词", "code": "yeoiav", "type": "Phrase", "weight": 100},
            ]
            async def words(requested):
                return {"success": True, "results": [
                    {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                    for word in requested
                ]}
            async def codes(requested):
                return {"success": True, "results": [
                    {"code": code, "phrases": [row for row in rows if row["code"] == code]}
                    for code in requested
                ]}
            with (
                patch.object(draft_tools, "_lookup_words_raw", side_effect=words),
                patch.object(draft_tools, "_lookup_codes_raw", side_effect=codes),
                patch.object(draft_tools, "_fetch_encode_candidates", AsyncMock(side_effect=AssertionError(
                    "an exact existing code weight change must not re-encode the item"
                ))),
                patch.object(draft_tools, "keytao_list_draft_items", AsyncMock(return_value={
                    "success": True, "batchId": "", "contentVersion": 0, "items": [],
                })),
            ):
                plan = await draft_tools._prepare_ranked_reorder_plan(
                    "qq", "s57-scope", ["嘢", "咽"], "yeoiav",
                    expected_codes=["yeoiav", "yeoiav"], expected_weights=[10, 11],
                )
            self.assertTrue(plan.get("success"), plan)
            self.assertEqual(
                [(item["word"], item["code"], item["weight"]) for item in plan["items"]],
                [("嘢", "yeoiav", 10), ("咽", "yeoiav", 11)],
            )
        asyncio.run(run())

    def test_incident_request_creates_real_single_weight_ticket(self):
        async def run():
            user = "s57-reorder"
            rows = [
                {"word": "咽", "code": "yeoiav", "type": "Single", "weight": 10},
                {"word": "嘢", "code": "yeoiav", "type": "Single", "weight": 11},
            ]
            snapshot = {"success": True, "batchId": "", "contentVersion": 0, "items": []}

            async def words(requested):
                return {"success": True, "results": [
                    {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                    for word in requested
                ]}

            async def codes(requested):
                return {"success": True, "results": [
                    {"code": code, "phrases": [row for row in rows if row["code"] == code]}
                    for code in requested
                ]}

            async def call(name, args, *_args, **_kwargs):
                self.assertIn(name, {
                    **harness._lookup_tools.TOOL_FUNCTIONS,
                    **draft_tools.TOOL_FUNCTIONS,
                }, "route dispatched an unregistered production tool")
                if name == "keytao_lookup_by_words_batch":
                    result = await words(args["words"])
                elif name == "keytao_lookup_by_code":
                    result = {"success": True, "phrases": rows}
                elif name == "keytao_list_draft_items":
                    result = snapshot
                elif name == "keytao_shift_phrase_code":
                    result = await draft_tools.keytao_shift_phrase_code("qq", user, **args)
                else:
                    raise AssertionError(name)
                return json.dumps(result, ensure_ascii=False)

            with (
                patch.object(commands, "call_tool_function", side_effect=call),
                patch.object(draft_tools, "_lookup_words_raw", side_effect=words),
                patch.object(draft_tools, "_lookup_codes_raw", side_effect=codes),
                patch.object(draft_tools, "keytao_list_draft_items", AsyncMock(return_value=snapshot)),
                patch.object(draft_tools, "_fetch_encode_candidates", AsyncMock(return_value={
                    "success": True, "candidateCodes": ["ye"],
                })),
                patch.object(draft_tools, "_keytao_strict_batch_add_to_draft", AsyncMock(return_value={
                    "success": False, "requiresConfirmation": True,
                    "warningDigest": "d" * 64, "warnings": [],
                })),
                patch("keytao_bot.utils.keytao_review._query_commonness_reference", side_effect=lambda word: {
                    "available": True, "attested": True, "corpusFrequency": 1000 if word == "咽" else 10,
                    "dictionaryPresenceCount": 2,
                }),
            ):
                reply = await commands._try_handle_draft_management_command(
                    '提高"嘢    yeoiav"的优先级，使之位于"咽    yeoiav"之前',
                    "qq", user, command_intent=commands.MessageCommandIntent(),
                )
            self.assertIsNotNone(reply, "explicit priority request must create a deterministic preview")
            record = chat.conversation_state_store.get_record(("qq", user))
            self.assertIsNotNone(record, reply)
            self.assertTrue(commands.server_warning_ticket_is_complete(record.state), reply)
            plan = record.state.args["_pending_display"]["shiftPlan"]
            self.assertEqual(
                [(item["word"], item["weight"], item["type"]) for item in plan["proposedState"]],
                [("嘢", 10, "Single"), ("咽", 11, "Single")],
            )
            self.assertIn("常用度提示：咽 更常用，仍按你的要求执行", reply)
            self.assertTrue(chat._advertised_reply_matches_live_record(reply, record), reply)
            delivered = chat._enforce_advertised_reply_contract(reply, ("qq", user))
            self.assertIn("常用度提示：咽 更常用，仍按你的要求执行", delivered)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
