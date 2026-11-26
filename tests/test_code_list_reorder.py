"""Offline regression checks for multi-code reorder across deterministic and semantic routes."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from tests import test_state_machine as harness
from tests.test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.operation_request import OperationRequest, understand_message
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingToolConfirm,
)
from keytao_bot.skills import __name__ as _skills_pkg  # noqa: F401
from keytao_bot.utils import keytao_review


_DRAFT_TOOLS = harness._draft_tools


class CodeListReorderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.routing = harness.chat_routing_module
        self.author = ConversationAddress.group("qq", "code-list-reorder-group", "rea")
        self.other = ConversationAddress.group("qq", "code-list-reorder-group", "other-user")
        self.store = MemoryConversationStateStore()
        self.client = FakeClient()
        self.rows = [
            {"word": "零度天", "code": "jdt", "type": "Phrase", "weight": 100},
            {"word": "一等天", "code": "jdtv", "type": "Phrase", "weight": 100},
            {"word": "二等天", "code": "jdtvo", "type": "Phrase", "weight": 100},
        ]
        self.candidates = {
            "零度天": ["jdt", "jdtv", "jdtvo", "jdtvoi"],
            "一等天": ["jdt", "jdtv", "jdtvo", "jdtvoi"],
            "二等天": ["jdt", "jdtv", "jdtvo", "jdtvoi"],
        }
        self.draft_items = []
        self.draft_version = 3
        self.calls = []
        self.writes = []
        self.ranking_override = None
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "reorder,direct_exec",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.commands, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.commands, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", side_effect=self.call),
            patch.object(self.commands, "call_tool_function", side_effect=self.call),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "_finish_ai_chat_response", new_callable=AsyncMock),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    async def _lookup_words_raw(self, words):
        return {
            "success": True,
            "results": [
                {"word": word, "phrases": [dict(row) for row in self.rows if row["word"] == word]}
                for word in words
            ],
        }

    async def _lookup_codes_raw(self, codes):
        return {
            "success": True,
            "results": [
                {
                    "success": True,
                    "code": code,
                    "phrases": [dict(row) for row in self.rows if row["code"] == code],
                }
                for code in codes
            ],
        }

    async def _fetch_encode_candidates(self, word, requested_code=None):
        codes = list(self.candidates.get(word, []))
        return {"success": bool(codes), "word": word, "candidateCodes": codes}

    async def _list_draft_items(self, platform, platform_id, **_kwargs):
        return {
            "success": True,
            "batchId": "code-list-batch" if self.draft_items else "",
            "contentVersion": self.draft_version if self.draft_items else 0,
            "items": copy.deepcopy(self.draft_items),
        }

    async def _strict_batch_add(self, platform, platform_id, items, **kwargs):
        if kwargs.get("confirmed"):
            self.writes.append(copy.deepcopy(items))
            return {
                "success": True,
                "batchId": "code-list-batch",
                "batchUrl": "https://offline.invalid/draft/code-list-batch",
                "contentVersion": self.draft_version + 1,
                "successCount": len(items),
                "failedCount": 0,
            }
        return {
            "success": False,
            "requiresConfirmation": True,
            "batchId": "code-list-batch",
            "batchUrl": "https://offline.invalid/draft/code-list-batch",
            "contentVersion": self.draft_version if self.draft_items else 0,
            "warningDigest": "a" * 64,
            "warnings": [],
            "failedCount": 0,
            "skippedCount": 0,
        }

    async def call(self, name, args, platform, user_id, **kwargs):
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_lookup_by_codes_batch":
            data = await self._lookup_codes_raw(args["codes"])
        elif name == "keytao_lookup_by_code":
            code = args["code"]
            data = {
                "success": True,
                "code": code,
                "phrases": [dict(row) for row in self.rows if row["code"] == code],
            }
        elif name == "keytao_lookup_by_words_batch":
            data = await self._lookup_words_raw(args["words"])
        elif name == "keytao_list_draft_items":
            data = await self._list_draft_items(platform, user_id)
        elif name == "keytao_encode":
            word = args["word"]
            codes = list(self.candidates.get(word, []))
            data = {"success": bool(codes), "word": word, "code": codes[0] if codes else "", "candidateCodes": codes}
        elif name == "keytao_shift_phrase_code":
            with (
                patch.object(_DRAFT_TOOLS, "_fetch_encode_candidates", side_effect=self._fetch_encode_candidates),
                patch.object(_DRAFT_TOOLS, "_lookup_words_raw", side_effect=self._lookup_words_raw),
                patch.object(_DRAFT_TOOLS, "_lookup_codes_raw", side_effect=self._lookup_codes_raw),
                patch.object(_DRAFT_TOOLS, "keytao_list_draft_items", side_effect=self._list_draft_items),
                patch.object(_DRAFT_TOOLS, "_keytao_strict_batch_add_to_draft", side_effect=self._strict_batch_add),
            ):
                data = await _DRAFT_TOOLS.keytao_shift_phrase_code(platform, user_id, **args)
        else:
            raise AssertionError(f"Unexpected tool call: {name} {args!r}")
        return json.dumps(data, ensure_ascii=False)

    async def run_semantic_stage(self, message, raw_op):
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_op, ensure_ascii=False,
        )
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            normalized_message_text=message, conv_key=self.author,
            space_key=self.author.space_key, owner_label="Rea",
        )
        handled = await self.chat._stage_understand_operation(ctx)
        return handled, ctx.response

    async def answer(self, message, actor=None):
        actor = actor or self.author
        return await self.commands.handle_pending_message_core(
            message, "qq", actor.actor_id, actor,
            space_key=actor.space_key, allow_intent_model=False,
        )

    def test_code_list_reorder_parser_accepts_natural_forms_and_rejects_unsafe_frames(self):
        parse_list = self.routing._parse_code_list_reorder_command
        parse_chain = self.routing._parse_code_chain_reorder_command

        accepted = {
            "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条": ("jdt", "jdtv", "jdtvo"),
            "重新排序 jdt jdtv jdtvo 三个编码的词条": ("jdt", "jdtv", "jdtvo"),
            "重新排序 jdt jdtv jdtvo": ("jdt", "jdtv", "jdtvo"),
            "重新排序下 jdt jdtv jdtvo": ("jdt", "jdtv", "jdtvo"),
            "重排 jdt、jdtv、jdtvo 这三个编码的词条": ("jdt", "jdtv", "jdtvo"),
            "重排 jdt，jdtv，jdtvo 3个编码下的词条": ("jdt", "jdtv", "jdtvo"),
            "重新排序 jdt、jdtv 和 jdtvo 三个编码的词条": ("jdt", "jdtv", "jdtvo"),
            "把 jdt jdtv jdtvo 三个编码的词条重新排序": ("jdt", "jdtv", "jdtvo"),
            "把 jdt、jdtv、jdtvo 这几个编码的词条按常用度排一下": ("jdt", "jdtv", "jdtvo"),
            "按常用度重新排序 jdt jdtv jdtvo 三个编码的词条": ("jdt", "jdtv", "jdtvo"),
            "jdt jdtv jdtvo 三个编码的词条重新排序": ("jdt", "jdtv", "jdtvo"),
            "请帮我重新排序 JDT JDTV JDTVO 三个编码的词条。": ("jdt", "jdtv", "jdtvo"),
            "重新排序\njdt\njdtv\njdtvo": ("jdt", "jdtv", "jdtvo"),
            "重新排序 ye yo": ("ye", "yo"),
        }
        for message, expected in accepted.items():
            with self.subTest(message=message):
                parsed = parse_list(message)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.codes, expected)
                self.assertIsNone(parse_chain(message), "multi-code must never glue into single code")
                understood = understand_message(message)
                self.assertEqual(understood.op, "reorder")
                self.assertEqual(understood.operands, ())
                self.assertEqual(understood.codes, expected)

        for single_msg, code in (
            ("重新排序 mkdr 编码的词条", "mkdr"),
            ("重排 mkdr 这个编码的词条", "mkdr"),
            ("把 mkdr 的词条重新排序", "mkdr"),
        ):
            with self.subTest(single_msg=single_msg):
                self.assertIsNone(parse_list(single_msg))
                parsed = parse_chain(single_msg)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.code, code)

        for rejected in (
            "不要重新排序 jdt jdtv jdtvo",
            "别重排 jdt jdtv jdtvo",
            "重新排序 jdt jdtv jdtvo 吗？",
            "他说重新排序 jdt jdtv jdtvo",
            "引用：重新排序 jdt jdtv jdtvo",
            "「重新排序 jdt jdtv jdtvo」",
            "重新排序 jdt jdtv jdtvo 然后提交",
            "重新排序 jdt jdt",
            "重新排序 jdt jdtv7",
            "重新排序 jdt abcdefg",
            "重新排序 jdt",
            "jdt jdtv jdtvo",
            "重新排序 米等 幂等 迷瞪",
        ):
            with self.subTest(rejected=rejected):
                self.assertIsNone(parse_list(rejected))

        rendered = self.commands.render_understood_command(
            OperationRequest(op="reorder", codes=("jdt", "jdtv", "jdtvo")),
        )
        self.assertEqual(rendered, "重新排序 jdt jdtv jdtvo")
        self.assertEqual(
            understand_message(rendered).codes,
            ("jdt", "jdtv", "jdtvo"),
        )
        pair_chain = self.commands.render_understood_command(
            OperationRequest(op="reorder", operands=("茂才", "冒菜"), codes=("mkdr",)),
        )
        self.assertEqual(pair_chain, "调整 mkdr 茂才 和 冒菜")

    async def test_verbatim_production_command_reorders_and_confirms_via_semantic_direct(self):
        ranking = {
            "status": "reorder",
            "reason": "comparison_edges",
            "currentOrder": list(self.rows),
            "proposedOrder": [self.rows[1], self.rows[2], self.rows[0]],
            "comparisons": [{"summary": "「一等天」较「零度天」更常用：语料频次 800 vs 20"}],
            "evidenceLines": [
                "「一等天」：语料频次 800，词典收录 3",
                "「二等天」：语料频次 300，词典收录 2",
                "「零度天」：语料频次 20，词典收录 1",
            ],
        }
        with patch.object(
            self.commands.keytao_review,
            "rank_code_chain_by_commonness",
            AsyncMock(return_value=ranking),
        ):
            handled, response = await self.run_semantic_stage(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
            )
        self.assertTrue(handled)
        self.assertNotIn("看到了操作请求", response)
        self.assertNotIn("还没有核对出可执行", response)
        self.assertIn("一等天：jdtv / 100（词库）→ jdt / 100", response)
        self.assertIn("二等天：jdtvo / 100（词库）→ jdtv / 100", response)
        self.assertIn("零度天：jdt / 100（词库）→ jdtvo / 100", response)
        self.assertIn("回复「确认」执行", response)
        self.assertEqual(self.writes, [])

        record = self.store.get_record(self.author)
        self.assertIsNotNone(record)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.state.function_name, "keytao_shift_phrase_code")
        self.assertEqual(record.state.confirmation_source, "server_warning")

        # Another actor cannot consume the sealed reorder ticket.
        self.assertIsNone(await self.answer("确认", self.other))
        self.assertEqual(self.writes, [])

        confirmed = await self.answer("确认")
        self.assertEqual(len(self.writes), 1, confirmed)
        self.assertIn("✅ 操作已完成", confirmed)
        self.assertEqual(
            {(item["action"], item["word"], item["code"]) for item in self.writes[0]},
            {
                ("Delete", "一等天", "jdtv"), ("Create", "一等天", "jdt"),
                ("Delete", "二等天", "jdtvo"), ("Create", "二等天", "jdtv"),
                ("Delete", "零度天", "jdt"), ("Create", "零度天", "jdtvo"),
            },
        )
        self.assertIsNone(self.store.get_record(self.author))

    async def test_stray_prose_operand_and_paraphrase_and_confirm_mode_all_execute(self):
        ranking = {
            "status": "reorder",
            "reason": "comparison_edges",
            "currentOrder": list(self.rows),
            "proposedOrder": [self.rows[1], self.rows[0], self.rows[2]],
            "comparisons": [{"summary": "「一等天」较「零度天」更常用"}],
            "evidenceLines": ["「一等天」：语料频次 800", "「零度天」：语料频次 20"],
        }
        with patch.object(
            self.commands.keytao_review,
            "rank_code_chain_by_commonness",
            AsyncMock(return_value=ranking),
        ):
            # Case 1: Model copies generic noun "词条" alongside exact codes.
            handled, response = await self.run_semantic_stage(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                raw_operation(op="reorder", operands=["词条"], codes=["jdt", "jdtv", "jdtvo"]),
            )
            self.assertTrue(handled)
            self.assertIn("一等天：jdtv / 100（词库）→ jdt / 100", response)
            self.assertNotIn("还没有核对出可执行", response)
            self.store.delete(self.author)

            # Case 2: Free-form paraphrase not matching strict regex.
            handled, response = await self.run_semantic_stage(
                "帮我把 jdt、jdtv、jdtvo 这三个编码里的词排个序",
                raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
            )
            self.assertTrue(handled)
            self.assertIn("一等天：jdtv / 100（词库）→ jdt / 100", response)
            self.store.delete(self.author)

            # Case 3: Semantic route without direct_exec seals understood command then shift ticket.
            os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "reorder"
            handled, prompt = await self.run_semantic_stage(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
            )
            self.assertTrue(handled)
            self.assertIn("重新排序 jdt jdtv jdtvo", prompt)
            self.assertIn("回复「确认」执行", prompt)
            plan_prompt = await self.answer("确认")
            self.assertIn("一等天：jdtv / 100（词库）→ jdt / 100", plan_prompt)
            done = await self.answer("确认")
            self.assertIn("✅ 操作已完成", done)
            self.assertEqual(len(self.writes), 1)

    async def test_legacy_deterministic_draft_management_handles_verbatim_command_without_semantic_route(self):
        ranking = {
            "status": "reorder",
            "reason": "comparison_edges",
            "currentOrder": list(self.rows),
            "proposedOrder": [self.rows[1], self.rows[0], self.rows[2]],
            "comparisons": [{"summary": "「一等天」较「零度天」更常用"}],
            "evidenceLines": ["「一等天」：语料频次 800", "「零度天」：语料频次 20"],
        }
        intent = self.routing._structural_draft_management_intent(
            "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
        )
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent, "code_list_reorder")
        self.assertEqual(intent.requested_codes, ("jdt", "jdtv", "jdtvo"))
        self.assertTrue(
            self.routing._is_fresh_current_user_command_intent(
                intent, "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
            )
        )
        with patch.object(
            self.commands.keytao_review,
            "rank_code_chain_by_commonness",
            AsyncMock(return_value=ranking),
        ):
            response = await self.commands._try_handle_draft_management_command(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                "qq",
                self.author.actor_id,
                self.author.space_key,
                "Rea",
                command_intent=intent,
            )
        self.assertIsNotNone(response)
        self.assertIn("一等天：jdtv / 100（词库）→ jdt / 100", response)
        self.assertNotIn("看到了操作请求", response)

    async def test_distinct_code_ranking_uses_code_length_order_not_unicode_order(self):
        # Unicode order is 一等天 < 二等天 < 零度天, opposite to current chain jdt < jdtv < jdtvo.
        strength = {"零度天": 3, "一等天": 2, "二等天": 1}

        async def compare(front, behind, **_kwargs):
            verdict = "front_more_common" if strength[front] > strength[behind] else "behind_more_common"
            return {
                "success": True,
                "verdict": verdict,
                "decisionReason": "bcc_frequency_ratio",
                "summary": f"{front} vs {behind}",
                "front": {"reference": {"attested": True, "corpusFrequency": strength[front] * 100}},
                "behind": {"reference": {"attested": True, "corpusFrequency": strength[behind] * 100}},
            }

        with patch.object(keytao_review, "compare_word_commonness", side_effect=compare):
            already = await keytao_review.rank_code_chain_by_commonness(self.rows)
        self.assertEqual(already["status"], "already_ordered")
        self.assertEqual([row["word"] for row in already["currentOrder"]], ["零度天", "一等天", "二等天"])

        with patch.object(keytao_review, "compare_word_commonness", side_effect=compare):
            handled, noop_reply = await self.run_semantic_stage(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
            )
        self.assertTrue(handled)
        self.assertIn("编码 jdt 顺序不变：零度天(jdt) → 一等天(jdtv) → 二等天(jdtvo)", noop_reply)
        self.assertIsNone(self.store.get_record(self.author))

        # Flip strengths so 一等天 > 二等天 > 零度天: must detect reorder, NOT already_ordered!
        strength = {"一等天": 3, "二等天": 2, "零度天": 1}
        with patch.object(keytao_review, "compare_word_commonness", side_effect=compare):
            needs_reorder = await keytao_review.rank_code_chain_by_commonness(self.rows)
        self.assertEqual(needs_reorder["status"], "reorder")
        self.assertEqual([row["word"] for row in needs_reorder["proposedOrder"]], ["一等天", "二等天", "零度天"])

    async def test_branching_chain_and_unrelated_extra_code_are_planned_cleanly(self):
        # 零度天 branches to jdta (not jdtv) AND has an unrelated extra code fdt outside jdt*.
        self.rows.append({"word": "零度天", "code": "fdt", "type": "Phrase", "weight": 100})
        self.candidates["零度天"] = ["jdt", "jdta", "jdtao"]
        with (
            patch.object(_DRAFT_TOOLS, "_fetch_encode_candidates", side_effect=self._fetch_encode_candidates),
            patch.object(_DRAFT_TOOLS, "_lookup_words_raw", side_effect=self._lookup_words_raw),
            patch.object(_DRAFT_TOOLS, "_lookup_codes_raw", side_effect=self._lookup_codes_raw),
            patch.object(_DRAFT_TOOLS, "keytao_list_draft_items", side_effect=self._list_draft_items),
        ):
            plan = await _DRAFT_TOOLS._prepare_ranked_reorder_plan(
                "qq", self.author.actor_id, ["一等天", "二等天", "零度天"], "jdt",
            )
        self.assertTrue(plan.get("success"), plan)
        self.assertEqual(
            [(entry["word"], entry["code"]) for entry in plan["proposedState"]],
            [("一等天", "jdt"), ("二等天", "jdtv"), ("零度天", "jdta")],
        )

    async def test_fail_closed_domain_refusals_explain_exact_cause_without_generic_block(self):
        # 1. Empty named code
        self.rows = [row for row in self.rows if row["code"] != "jdtvo"]
        handled, reply = await self.run_semantic_stage(
            "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
            raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
        )
        self.assertTrue(handled)
        self.assertIn("jdtvo", reply)
        self.assertIn("没有词条", reply)
        self.assertNotIn("看到了操作请求", reply)

        # 2. Unrelated candidate chains
        self.rows = [
            {"word": "零度天", "code": "jdt", "type": "Phrase", "weight": 100},
            {"word": "香蕉", "code": "xj", "type": "Phrase", "weight": 100},
        ]
        self.candidates["香蕉"] = ["xj", "xjo"]
        handled, reply = await self.run_semantic_stage(
            "重新排序 jdt xj",
            raw_operation(op="reorder", operands=[], codes=["jdt", "xj"]),
        )
        self.assertTrue(handled)
        self.assertIn("不在同一候选前缀链", reply)
        self.assertNotIn("看到了操作请求", reply)

        # 3. Mixed phrase types
        self.rows = [
            {"word": "甲", "code": "jdt", "type": "Single", "weight": 10},
            {"word": "一等天", "code": "jdtv", "type": "Phrase", "weight": 100},
        ]
        handled, reply = await self.run_semantic_stage(
            "重新排序 jdt jdtv",
            raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv"]),
        )
        self.assertTrue(handled)
        self.assertIn("词条类型不同", reply)
        self.assertNotIn("看到了操作请求", reply)

    async def test_draft_entry_is_merged_into_code_list_reorder_view(self):
        self.rows = [row for row in self.rows if row["code"] != "jdtvo"]
        self.draft_items = [
            {"id": 501, "action": "Create", "word": "二等天", "code": "jdtvo", "type": "Phrase", "weight": 100},
        ]
        ranking = {
            "status": "reorder",
            "reason": "comparison_edges",
            "currentOrder": [
                {"word": "零度天", "code": "jdt", "type": "Phrase", "weight": 100, "source": "live"},
                {"word": "一等天", "code": "jdtv", "type": "Phrase", "weight": 100, "source": "live"},
                {"word": "二等天", "code": "jdtvo", "type": "Phrase", "weight": 100, "source": "draft"},
            ],
            "proposedOrder": [
                {"word": "一等天", "code": "jdtv", "type": "Phrase", "weight": 100, "source": "live"},
                {"word": "零度天", "code": "jdt", "type": "Phrase", "weight": 100, "source": "live"},
                {"word": "二等天", "code": "jdtvo", "type": "Phrase", "weight": 100, "source": "draft"},
            ],
            "comparisons": [{"summary": "「一等天」较「零度天」更常用"}],
            "evidenceLines": ["「一等天」：语料频次 800", "「零度天」：语料频次 20"],
        }
        with patch.object(
            self.commands.keytao_review,
            "rank_code_chain_by_commonness",
            AsyncMock(return_value=ranking),
        ):
            handled, response = await self.run_semantic_stage(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
            )
        self.assertTrue(handled)
        self.assertIn("一等天：jdtv / 100（词库）→ jdt / 100", response)
        self.assertIn("零度天：jdt / 100（词库）→ jdtv / 100", response)
        self.assertNotIn("没有词条", response)

    async def test_full_delivery_guard_preserves_code_list_reorder_confirmation_ticket(self):
        ranking = {
            "status": "reorder",
            "reason": "comparison_edges",
            "currentOrder": list(self.rows),
            "proposedOrder": [self.rows[1], self.rows[2], self.rows[0]],
            "comparisons": [{"summary": "「一等天」较「零度天」更常用"}],
            "evidenceLines": ["「一等天」：语料频次 800", "「零度天」：语料频次 20"],
        }
        with patch.object(
            self.commands.keytao_review,
            "rank_code_chain_by_commonness",
            AsyncMock(return_value=ranking),
        ):
            handled, response = await self.run_semantic_stage(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
            )
        self.assertTrue(handled)
        token = self.chat._current_turn_message.set("喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条")
        try:
            guarded = self.chat._enforce_advertised_reply_contract(response, self.author)
        finally:
            self.chat._current_turn_message.reset(token)
        self.assertIn("一等天：jdtv / 100（词库）→ jdt / 100", guarded)
        self.assertIn("回复「确认」执行", guarded)
        self.assertNotIn("没找到要执行的操作", guarded)
        self.assertNotIn("看到了操作请求", guarded)

    async def test_ranked_plan_shows_unchanged_member_and_all_bcc_evidence(self):
        self.rows = [
            {"word": "超导体", "code": "jdt", "type": "Phrase", "weight": 100},
            {"word": "基督徒", "code": "jdtv", "type": "Phrase", "weight": 100},
            {"word": "进度条", "code": "jdtvo", "type": "Phrase", "weight": 100},
        ]
        self.candidates = {
            "超导体": ["jdt", "jdtv", "jdtva", "jdtvai"],
            "基督徒": ["jdt", "jdtv", "jdtvi", "jdtviu"],
            "进度条": ["jdt", "jdtv", "jdtvo", "jdtvou"],
        }
        ranking = {
            "status": "reorder",
            "reason": "comparison_edges",
            "currentOrder": list(self.rows),
            "proposedOrder": [self.rows[1], self.rows[0], self.rows[2]],
            "comparisons": [{"summary": "「基督徒」较「超导体」更常用"}],
            "evidenceLines": [
                "「超导体」：BCC 多领域 176（每百万 0.44）；新闻 795（每百万 0.86）；词典收录 3",
                "「基督徒」：BCC 多领域 1,842（每百万 4.56）；新闻 771（每百万 0.83）；词典收录 2",
                "「进度条」：语料频次 3，词典收录 1；现代四频道均未收录；历史补充：古代汉语 未收录；近代汉语 未收录",
            ],
        }
        with patch.object(
            self.commands.keytao_review,
            "rank_code_chain_by_commonness",
            AsyncMock(return_value=ranking),
        ):
            handled, response = await self.run_semantic_stage(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
            )
        self.assertTrue(handled)
        self.assertIn("• 基督徒：jdtv / 100（词库）→ jdt / 100", response)
        self.assertIn("• 超导体：jdt / 100（词库）→ jdtv / 100", response)
        self.assertIn("• 进度条：jdtvo / 100（词库，保持不变）", response)
        self.assertIn(
            "依据：关键证据：基督徒 BCC 多领域 4.56/百万，超导体 BCC 多领域 0.44/百万，进度条 语料3/词典1（BCC未收录）",
            response,
        )
        self.assertIn(
            "如需手动排序，可引用本条回复「重新排序为 基督徒 > 超导体 > 进度条」（按你想要的顺序）。",
            response,
        )
        token = self.chat._current_turn_message.set("喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条")
        try:
            guarded = self.chat._enforce_advertised_reply_contract(response, self.author)
        finally:
            self.chat._current_turn_message.reset(token)
        self.assertIn("• 进度条：jdtvo / 100（词库，保持不变）", guarded)
        self.assertIn(
            "如需手动排序，可引用本条回复「重新排序为 基督徒 > 超导体 > 进度条」（按你想要的顺序）。",
            guarded,
        )

    def test_explicit_order_reorder_parser_accepts_natural_forms_and_rejects_unsafe_frames(self):
        parse_explicit = self.routing._parse_explicit_order_reorder_command
        parse_word_list = self.routing._parse_word_list_reorder_command

        accepted = {
            "重新排序为 进度条 > 基督徒 > 超导体": ("进度条", "基督徒", "超导体"),
            "喵喵 重新排序为 进度条 > 基督徒 > 超导体": ("进度条", "基督徒", "超导体"),
            "重新排序为：进度条 ＞ 基督徒 ＞ 超导体": ("进度条", "基督徒", "超导体"),
            "重排为 进度条 → 基督徒 → 超导体": ("进度条", "基督徒", "超导体"),
            "排序为 进度条 -> 基督徒 -> 超导体": ("进度条", "基督徒", "超导体"),
            "顺序改为 进度条、基督徒、超导体": ("进度条", "基督徒", "超导体"),
            "重新排序为 进度条 基督徒 超导体": ("进度条", "基督徒", "超导体"),
            "改为 进度条 > 基督徒 > 超导体": ("进度条", "基督徒", "超导体"),
            "按 进度条 > 基督徒 > 超导体 排序": ("进度条", "基督徒", "超导体"),
            "按 进度条 → 基督徒 → 超导体 重排": ("进度条", "基督徒", "超导体"),
            "进度条 > 基督徒 > 超导体": ("进度条", "基督徒", "超导体"),
            "进度条 → 基督徒 → 超导体": ("进度条", "基督徒", "超导体"),
            "进度条 > 超导体": ("进度条", "超导体"),
            "请帮我重新排序为 进度条 > 基督徒 > 超导体。": ("进度条", "基督徒", "超导体"),
        }
        for message, expected in accepted.items():
            with self.subTest(message=message):
                parsed = parse_explicit(message)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.words, expected)
                understood = understand_message(message)
                self.assertEqual(understood.op, "reorder")
                self.assertEqual(understood.operands, expected)
                self.assertEqual(understood.codes, ())
                self.assertEqual(understood.relation, "explicit_order")

        # Commonness-ranked word-list reorder must NOT be hijacked as explicit order.
        for commonness_msg in (
            "重新排序 米等 幂等 迷瞪",
            "重新排序下 米等 幂等 迷瞪",
            "米等 幂等 迷瞪 排一下",
        ):
            with self.subTest(commonness_msg=commonness_msg):
                self.assertIsNone(parse_explicit(commonness_msg))
                self.assertIsNotNone(parse_word_list(commonness_msg))

        for rejected in (
            "不要重新排序为 进度条 > 基督徒 > 超导体",
            "别改为 进度条 > 基督徒 > 超导体",
            "重新排序为 进度条 > 基督徒 > 超导体 吗？",
            "可以重新排序为 进度条 > 基督徒 > 超导体吗",
            "他说重新排序为 进度条 > 基督徒 > 超导体",
            "「重新排序为 进度条 > 基督徒 > 超导体」",
            "重新排序为 进度条 > 进度条 > 超导体",
            "重新排序为 进度条 > 基督徒 > 超导体 然后提交",
            "重新排序为 jdt > jdtv > jdtvo",
            "按 常用度 排序",
            "超导体 → 基督徒",
            "改为 春暖花开",
        ):
            with self.subTest(rejected=rejected):
                self.assertIsNone(parse_explicit(rejected))

        rendered = self.commands.render_understood_command(
            OperationRequest(
                op="reorder",
                operands=("进度条", "基督徒", "超导体"),
                relation="explicit_order",
            ),
        )
        self.assertEqual(rendered, "重新排序为 进度条 > 基督徒 > 超导体")
        self.assertEqual(
            understand_message(rendered).relation,
            "explicit_order",
        )

    async def test_quoted_live_plan_manual_reorder_auto_encodes_and_requires_reconfirmation(self):
        # Exact production setup: 超导体 has both jdt and fly-key wdtv!
        self.rows = [
            {"word": "超导体", "code": "jdt", "type": "Phrase", "weight": 100},
            {"word": "超导体", "code": "wdtv", "type": "Phrase", "weight": 100},
            {"word": "基督徒", "code": "jdtv", "type": "Phrase", "weight": 100},
            {"word": "进度条", "code": "jdtvo", "type": "Phrase", "weight": 100},
        ]
        self.candidates = {
            "超导体": ["jdt", "jdtv", "jdtva", "jdtvai", "wdt", "wdtv", "wdtva", "wdtvai"],
            "基督徒": ["jdt", "jdtv", "jdtvi", "jdtviu"],
            "进度条": ["jdt", "jdtv", "jdtvo", "jdtvou"],
        }
        ranking = {
            "status": "reorder",
            "reason": "comparison_edges",
            "currentOrder": [self.rows[0], self.rows[2], self.rows[3]],
            "proposedOrder": [self.rows[2], self.rows[0], self.rows[3]],
            "comparisons": [{"summary": "「基督徒」较「超导体」更常用"}],
            "evidenceLines": [
                "「基督徒」：BCC 多领域 1,842（每百万 4.56）；词典收录 2",
                "「超导体」：BCC 多领域 176（每百万 0.44）；词典收录 3",
                "「进度条」：语料频次 3，词典收录 1；现代四频道均未收录",
            ],
        }
        with patch.object(
            self.commands.keytao_review,
            "rank_code_chain_by_commonness",
            AsyncMock(return_value=ranking),
        ):
            handled, initial_plan = await self.run_semantic_stage(
                "喵喵 重新排序 jdt jdtv jdtvo 三个编码的词条",
                raw_operation(op="reorder", operands=[], codes=["jdt", "jdtv", "jdtvo"]),
            )
        self.assertTrue(handled)
        self.assertIn("如需手动排序，可引用本条回复「重新排序为 基督徒 > 超导体 > 进度条」", initial_plan)
        self.assertEqual(self.writes, [])

        # Step 2: User quotes the live plan and specifies manual order.
        # Even if the semantic model omits relation="explicit_order", validator promotes it!
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_operation(
                op="reorder",
                operands=["进度条", "基督徒", "超导体"],
                codes=[],
                relation="",
            ),
            ensure_ascii=False,
        )
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            normalized_message_text="重新排序为 进度条 > 基督徒 > 超导体",
            conv_key=self.author,
            space_key=self.author.space_key, owner_label="Rea",
            reply_reference=self.chat.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text=initial_plan, message_id="msg-plan-1",
            ),
        )
        with patch.dict(os.environ, {
            "KEYTAO_SEMANTIC_ROUTE_OPS": "reorder,direct_exec,pending_release,pending_answer",
            "KEYTAO_SEMANTIC_SHADOW": "0",
        }):
            handled = await self.chat._stage_understand_operation_early(ctx)
        self.assertTrue(handled)
        manual_plan = ctx.response
        self.assertEqual(self.writes, [], "manual reorder must ALWAYS ask for confirmation first")
        self.assertIn("• 进度条：jdtvo / 100（词库）→ jdt / 100", manual_plan)
        self.assertIn("• 基督徒：jdtv / 100（词库，保持不变）", manual_plan)
        self.assertIn("• 超导体：jdt / 100（词库）→ jdtva / 100", manual_plan)
        self.assertIn("依据：按你指定的顺序：进度条 > 基督徒 > 超导体", manual_plan)
        self.assertIn("回复「确认」执行，或「取消」。", manual_plan)

        # Delivery guard must accept the new plan AND its manual-order hint verbatim.
        token = self.chat._current_turn_message.set("重新排序为 进度条 > 基督徒 > 超导体")
        try:
            guarded = self.chat._enforce_advertised_reply_contract(manual_plan, self.author)
        finally:
            self.chat._current_turn_message.reset(token)
        self.assertIn("• 超导体：jdt / 100（词库）→ jdtva / 100", guarded)
        self.assertIn("回复「确认」执行，或「取消」。", guarded)

        # Step 3: User replies 确认 -> exact auto-encoded write executes, wdtv untouched!
        confirmed = await self.answer("确认")
        self.assertEqual(len(self.writes), 1, confirmed)
        self.assertIn("✅ 操作已完成", confirmed)
        self.assertEqual(
            {(item["action"], item["word"], item["code"]) for item in self.writes[0]},
            {
                ("Delete", "进度条", "jdtvo"), ("Create", "进度条", "jdt"),
                ("Delete", "超导体", "jdt"), ("Create", "超导体", "jdtva"),
            },
        )
        self.assertIsNone(self.store.get_record(self.author))

    async def test_quoted_expired_plan_and_occupied_destination_auto_encode_and_shift(self):
        # Ticket already expired (store is empty), AND jdtva is occupied by 占位词!
        self.rows = [
            {"word": "超导体", "code": "jdt", "type": "Phrase", "weight": 100},
            {"word": "超导体", "code": "wdtv", "type": "Phrase", "weight": 100},
            {"word": "基督徒", "code": "jdtv", "type": "Phrase", "weight": 100},
            {"word": "进度条", "code": "jdtvo", "type": "Phrase", "weight": 100},
            {"word": "占位词", "code": "jdtva", "type": "Phrase", "weight": 100},
        ]
        self.candidates = {
            "超导体": ["jdt", "jdtv", "jdtva", "jdtvai", "wdt", "wdtv"],
            "基督徒": ["jdt", "jdtv", "jdtvi", "jdtviu"],
            "进度条": ["jdt", "jdtv", "jdtvo", "jdtvou"],
            "占位词": ["jdt", "jdtv", "jdtva", "jdtvai"],
        }
        expired_quote = (
            "🔁 调整计划：\n"
            "• 基督徒：jdtv / 100（词库）→ jdt / 100\n"
            "• 超导体：jdt / 100（词库）→ jdtv / 100\n"
            "• 进度条：jdtvo / 100（词库，保持不变）\n"
            "回复「确认」执行，或「取消」。"
        )
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_operation(
                op="reorder",
                operands=["进度条", "基督徒", "超导体"],
                codes=[],
                relation="explicit_order",
            ),
            ensure_ascii=False,
        )
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            normalized_message_text="重新排序为 进度条 > 基督徒 > 超导体",
            conv_key=self.author,
            space_key=self.author.space_key, owner_label="Rea",
            reply_reference=self.chat.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text=expired_quote, message_id="expired-msg",
            ),
        )
        with patch.dict(os.environ, {
            "KEYTAO_SEMANTIC_ROUTE_OPS": "reorder,direct_exec,pending_release,pending_answer",
            "KEYTAO_SEMANTIC_SHADOW": "0",
        }):
            handled = await self.chat._stage_understand_operation_early(ctx)
        self.assertTrue(handled)
        self.assertEqual(self.writes, [])
        self.assertIn("• 进度条：jdtvo / 100（词库）→ jdt / 100", ctx.response)
        self.assertIn("• 超导体：jdt / 100（词库）→ jdtva / 100", ctx.response)
        self.assertIn("• 占位词：jdtva→jdtvai（顺延）", ctx.response)
        self.assertIn("回复「确认」执行，或「取消」。", ctx.response)

        confirmed = await self.answer("确认")
        self.assertEqual(len(self.writes), 1, confirmed)
        self.assertIn(("Delete", "占位词", "jdtva"), {(i["action"], i["word"], i["code"]) for i in self.writes[0]})
        self.assertIn(("Create", "占位词", "jdtvai"), {(i["action"], i["word"], i["code"]) for i in self.writes[0]})

    async def test_legacy_path_supersedes_live_reorder_ticket_and_handles_noop(self):
        self.rows = [
            {"word": "超导体", "code": "jdt", "type": "Phrase", "weight": 100},
            {"word": "基督徒", "code": "jdtv", "type": "Phrase", "weight": 100},
            {"word": "进度条", "code": "jdtvo", "type": "Phrase", "weight": 100},
        ]
        self.candidates = {
            "超导体": ["jdt", "jdtv", "jdtva", "jdtvai"],
            "基督徒": ["jdt", "jdtv", "jdtvi", "jdtviu"],
            "进度条": ["jdt", "jdtv", "jdtvo", "jdtvou"],
        }
        # Seed a live PendingToolConfirm ticket.
        self.store.set(
            self.author,
            PendingToolConfirm(
                function_name="keytao_shift_phrase_code",
                args={"word": "基督徒", "target_code": "jdt", "confirmed_plan_digest": "f" * 64, "expected_content_version": 0},
                confirmation_source="server_warning",
            ),
            space_key=self.author.space_key,
            owner_label="Rea",
        )
        msg = "重新排序为 进度条 > 基督徒 > 超导体"
        intent = self.routing._structural_draft_management_intent(msg)
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent, "explicit_order_reorder")

        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            message_text=msg, normalized_message_text=msg,
            conv_key=self.author, space_key=self.author.space_key, owner_label="Rea",
            current_pending_record=self.store.get_record(self.author),
            generic_command_intent=intent,
        )
        with patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": ""}):
            await self.chat._stage_arbitrate_active_operation(ctx)
            self.assertTrue(ctx.generic_intent_is_fresh_command)
            await self.chat._stage_execute_pending_state(ctx)
            self.assertIsNone(ctx.response)
            await self.chat._stage_handle_draft_management(ctx)
        self.assertIsNotNone(ctx.response)
        self.assertIn("• 进度条：jdtvo / 100（词库）→ jdt / 100", ctx.response)
        self.assertIn("• 超导体：jdt / 100（词库）→ jdtva / 100", ctx.response)
        self.assertEqual(self.writes, [])

        # Already in requested order -> clean no-op, no ticket staged.
        self.store.delete(self.author)
        noop_msg = "重新排序为 超导体 > 基督徒 > 进度条"
        noop_intent = self.routing._structural_draft_management_intent(noop_msg)
        noop_reply = await self.commands._try_handle_draft_management_command(
            noop_msg, "qq", self.author.actor_id, self.author.space_key, "Rea", command_intent=noop_intent,
        )
        self.assertIn("无需调整", noop_reply)
        self.assertIsNone(self.store.get_record(self.author))


if __name__ == "__main__":
    unittest.main()
