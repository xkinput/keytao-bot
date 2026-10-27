"""Offline regression for partially resolved model-first multi-word queries."""

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.state import MemoryConversationStateStore, PendingToolConfirm
from keytao_bot.utils import keytao_review
from keytao_bot.utils.pending_confirmation import render_remediation_reply


class HotfixMultiwordTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="offline-owner", space_type="group",
            space_id="offline-multiword", speaker_name="Offline owner",
        )
        self.author = self.memory.conversation_address
        self.client = FakeClient()
        self.tool = AsyncMock(side_effect=self.lookup)
        self.main = AsyncMock(side_effect=AssertionError("Unexpected model call"))
        self.finish = AsyncMock()
        self.unresolved = {"车损"}
        self.failure_mode = "code_unresolved"
        self.unresolved_pronunciations = []
        self.codes = {
            "没头没尾": ("mtmw", "mtmwa"), "国潮": ("gljz", "glwz"),
            "泥炭": ("nktf", "nktfa"), "棉麻": ("mmms", "mmmsa"),
            "车损": ("wesy", "wesya"),
        }
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "query", "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat, "get_ai_response_core", self.main),
            patch.object(self.chat, "get_history", return_value=[]),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "schedule_memory_compaction"),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
            patch.object(self.chat._user_resolver, "resolved_binding_for_notice", return_value=True),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        for variable, value in (
            (self.chat._current_conversational_turn, None),
            (self.chat.current_draft_delivery_claims, []),
            (self.chat._current_turn_message, ""),
        ):
            token = variable.set(value)
            self.addCleanup(variable.reset, token)

    @staticmethod
    def unresolved_message(word):
        return f"「{word}」没有通过逐字权威读音交叉校验的候选读音，暂不推荐编码"

    async def unresolved_review(self, word, result):
        if self.failure_mode == "code_unresolved":
            # Use the lookup producer and the review failure path, without I/O.
            encoded = harness._normalize_encode_response(word, {"success": False, "codes": []})
            self.assertEqual(encoded["message"], render_remediation_reply(
                "编码服务未能返回有效候选编码", command=f"加词 {word}", words=(word,),
            ))
            with (
                patch.object(keytao_review, "collect_pronunciation_evidence_limited", AsyncMock(return_value={})),
                patch.object(keytao_review, "fetch_keytao_encode", AsyncMock(return_value=encoded)),
                patch.object(keytao_review, "lookup_words", AsyncMock(return_value={})),
            ):
                review = await keytao_review.prepare_reviewed_word(
                    keytao_review.ReviewHttpConfig("https://offline.invalid", "offline"), word,
                )
            self.assertFalse(review["success"])
            self.assertEqual(review["reviewVerdictSite"], "code_unresolved")
            self.assertNotIn("pronunciationUnresolved", review)
            return review
        if self.failure_mode == "missing_reading":
            result["pronunciations"][0]["pinyin"] = ""
            return result
        result.update(
            pronunciationUnresolved=True, pronunciations=self.unresolved_pronunciations,
            recommendedCode="", message=self.unresolved_message(word),
        )
        return result

    async def lookup(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        if name == "keytao_pending_items_by_words":
            result = {"success": True, "complete": True, "items": []}
        elif name == "keytao_lookup_by_word":
            result = {"success": True, "phrases": []}
        elif name == "keytao_prepare_reviewed_add":
            word = args["word"]
            manual = word in {"国潮", "棉麻"}
            result = {
                "success": True, "word": word, "type": "Phrase",
                "recommendedCode": self.codes[word][0],
                "needsManualReview": manual, "requiresManualReview": manual,
                "preSubmitAudit": {
                    "success": True, "verdict": "needs_admin" if manual else "pass",
                    "autoApprove": not manual, "summary": "Offline pronunciation evidence",
                    "issues": [],
                },
                "pronunciations": [{
                    "pinyin": "offline reading", "recommendedCode": self.codes[word][0],
                    "sources": [{"source": "semantic_context" if manual else "offline"}],
                    "candidateStatuses": [
                        {"code": code, "occupied": False, "label": "空位", "words": []}
                        for code in self.codes[word]
                    ],
                }],
            }
            if word in self.unresolved:
                result = await self.unresolved_review(word, result)
        else:
            self.fail(f"Unexpected offline tool: {name}")
        return json.dumps(result, ensure_ascii=False)

    async def query(self, words):
        message = " ".join(words)
        self.chat._current_turn_message.set(message)
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_operation(op="query", operands=list(words), codes=[], frame="none"),
            ensure_ascii=False,
        )
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            message_text=message, normalized_message_text=message,
            conv_key=self.author, space_key=self.author.space_key,
            memory_context=self.memory, owner_label="Offline owner",
            command_intent_for=AsyncMock(return_value=self.chat.MessageCommandIntent()),
        )
        with patch.object(self.commands, "_prepare_multi_word_query", wraps=self.commands._prepare_multi_word_query) as prepare:
            self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
        prepare.assert_awaited_once()
        self.assertEqual(prepare.await_args.args[0], words)
        self.client.create.assert_awaited_once()
        self.main.assert_not_awaited()
        self.finish.assert_awaited_once()
        self.assertEqual([
            call.args[1]["word"] for call in self.tool.await_args_list
            if call.args[0] == "keytao_prepare_reviewed_add"
        ], list(words))
        return ctx.response, self.chat._prepare_user_facing_reply(ctx.response, self.memory)

    async def assert_candidates(self, words):
        raw, delivered = await self.query(words)
        self.assertNotIn("当前没有可验证的可执行操作，本次未写入。", delivered)
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        record = self.store.get_record(self.author)
        self.assertIsNotNone(record)
        self.assertIsInstance(record.state, PendingToolConfirm)
        resolved = [word for word in words if word not in self.unresolved]
        self.assertEqual(record.state.args["_query_words"], list(words))
        self.assertEqual([row["word"] for row in record.state.args["items"]], resolved)
        self.assertEqual([row["word"] for row in record.state.args["_candidate_scopes"]], resolved)
        for block in record.state.args["_query_other_blocks"]:
            self.assertFalse(self.chat.advertised_reply_contract(block).requires_live_state)
        self.assertTrue(self.chat._advertised_reply_matches_live_record(delivered, record))
        self.assertEqual(
            self.chat._render_live_batch_record(record).splitlines(), raw.splitlines(),
        )
        self.assertEqual(record.owner_key, self.author)
        for command in self.chat.advertised_reply_contract(delivered).command_suggestions:
            self.assertTrue(harness.chat_routing_module.message_authorizes_live_pending_mutation(command, record.state), command)
        selection = "，".join(f"{word} 1" for word in resolved)
        selected, intent, error = self.chat._resolve_multi_word_pending_candidate_selection(record.state, selection)
        self.assertIsNone(error)
        self.assertIsNotNone(intent)
        self.assertEqual([row["word"] for row in selected.args["items"]], resolved)
        for word in words:
            if word in self.unresolved:
                line = f"「{word}」未能确认读音或候选编码，请提供读音或具体编码。"
                self.assertEqual(delivered.splitlines().count(line), 1)
                self.assertNotIn(f"「{word}」候选编码", delivered)
            else:
                self.assertIn(f"「{word}」", delivered)
                for code in self.codes[word]:
                    self.assertIn(code, delivered)
        return delivered

    async def test_four_resolved_and_one_unresolved(self):
        await self.assert_candidates(("没头没尾", "国潮", "泥炭", "棉麻", "车损"))

    async def test_one_resolved_and_one_unresolved(self):
        await self.assert_candidates(("国潮", "车损"))

    async def test_unresolved_before_resolved(self):
        await self.assert_candidates(("车损", "国潮"))

    async def test_missing_candidate_reading_remediation(self):
        self.failure_mode = "missing_reading"
        await self.assert_candidates(("国潮", "车损"))

    async def test_two_failed_words_have_one_short_line_each(self):
        self.unresolved = {"车损", "棉麻"}
        await self.assert_candidates(("车损", "国潮", "棉麻"))

    async def test_all_resolved_unchanged(self):
        self.unresolved.clear()
        delivered = await self.assert_candidates(("没头没尾", "国潮", "泥炭", "棉麻", "车损"))
        state = self.store.get(self.author)
        self.assertEqual(delivered, self.commands.render_server_backed_batch_candidates(
            state.args["items"], state.args["_candidate_scopes"],
        ))

    async def test_all_unresolved_keeps_per_word_explanations(self):
        self.unresolved = {"国潮", "车损"}
        self.failure_mode = "pronunciation_unresolved"
        self.unresolved_message = lambda word: (
            f"「{word}」的权威整词或逐字读音服务暂不可用，当前读音无法完成交叉验证，暂不推荐编码"
        )
        raw, delivered = await self.query(("国潮", "车损"))
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        for word in self.unresolved:
            self.assertIn(self.unresolved_message(word), delivered)
        self.assertNotIn("候选编码", delivered)
        self.assertIsNone(self.store.get_record(self.author))

    async def test_all_unresolved_preserves_existing_plain_explanations(self):
        self.unresolved = {"国潮", "车损"}
        self.failure_mode = "pronunciation_unresolved"
        self.unresolved_message = lambda word: f"「{word}」未取得能确认整词读音的证据，暂不推荐编码。"
        raw, delivered = await self.query(("国潮", "车损"))
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        for word in self.unresolved:
            self.assertIn(self.unresolved_message(word), delivered)
        self.assertIsNone(self.store.get_record(self.author))

    async def test_no_resolved_scope_preserves_original_remediation_blocks(self):
        self.unresolved = {"国潮", "车损"}
        response = await self.commands._prepare_multi_word_query(
            ("国潮", "车损"), "qq", self.author.actor_id, self.author,
            self.author.space_key, "Offline owner",
        )
        self.assertEqual(response, "\n\n".join(
            render_remediation_reply(
                "编码服务未能返回有效候选编码", command=f"加词 {word}", words=(word,),
            ) for word in ("国潮", "车损")
        ))
        self.assertIsNone(self.store.get_record(self.author))

    async def test_mixed_query_preserves_plain_unresolved_explanation(self):
        self.failure_mode = "pronunciation_unresolved"
        raw, delivered = await self.query(("国潮", "车损"))
        self.assertIn(self.unresolved_message("车损"), delivered)
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        record = self.store.get_record(self.author)
        self.assertEqual([row["word"] for row in record.state.args["items"]], ["国潮"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(delivered, record))

    async def test_real_multi_sense_message_with_reading_and_code_lines(self):
        self.failure_mode = "pronunciation_unresolved"
        self.unresolved = {"出圈"}
        self.unresolved_pronunciations = [
            {"pinyin": "chu juan", "codes": ["jjjt", "jjjto"]},
            {"pinyin": "chū quān", "codes": ["jjqt", "jjqto"]},
        ]
        self.codes["出圈"] = ("jjjt", "jjqt")
        self.unresolved_message = lambda word: (
            f"「{word}」存在含义不同的多个读音，当前含义与常用度证据"
            "未能唯一支持其中一个；本次不推荐编码，也不会创建待确认加词操作。\n"
            "- chu juan：jjjt、jjjto\n- chū quān：jjqt、jjqto\n"
            "请明确要采用的读音或具体含义。"
        )
        raw, delivered = await self.query(("国潮", "出圈"))
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        self.assertIn(self.unresolved_message("出圈"), delivered)
        record = self.store.get_record(self.author)
        self.assertEqual([row["word"] for row in record.state.args["items"]], ["国潮"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(delivered, record))

    async def test_real_requested_reading_mismatch_message(self):
        self.failure_mode = "pronunciation_unresolved"
        self.unresolved = {"出圈"}
        self.codes["出圈"] = ("jjjt", "jjqt")
        self.unresolved_message = lambda word: (
            f"「{word}」的指定读音 chu qiong 与编码服务返回的"
            "候选读音都不匹配。可用读音：chū juàn、chū quān。\n"
            "- chū juàn：jjjt、jjjto\n- chū quān：jjqt、jjqto\n"
            "请明确要采用的读音或具体含义；本次未写入。"
        )
        raw, delivered = await self.query(("出圈", "国潮"))
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        self.assertIn(self.unresolved_message("出圈"), delivered)
        record = self.store.get_record(self.author)
        self.assertEqual([row["word"] for row in record.state.args["items"]], ["国潮"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(delivered, record))


if __name__ == "__main__":
    unittest.main()
