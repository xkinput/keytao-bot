"""S54 routing and delivery regressions using the offline chat harness."""

import asyncio
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import SQLiteConversationStateStore, server_warning_pending_state

chat = harness.openai_chat_module
commands = harness.chat_commands_module


def review_payload(word):
    code, reading = {
        "大端": ("dsdt", "dà duān"),
        "小端": ("xcdt", "xiǎo duān"),
        "肌群": ("jkqt", "jī qún"),
    }[word]
    codes = [code, code + "v", code + "vo"]
    return {
        "success": True, "word": word, "recommendedCode": codes[-1],
        "autoReviewable": False, "needsManualReview": True,
        "manualReviewReason": "常用度证据不足",
        "preSubmitAudit": {"summary": "常用度证据不足"},
        "pronunciations": [{
            "pinyin": reading, "recommendedCode": codes[-1],
            "sources": [{"source": "汉典"}],
            "candidateStatuses": [
                {"code": candidate, "occupied": index < 2,
                 "words": ["占位词"] if index < 2 else [],
                 "label": "已有「占位词」" if index < 2 else "空位"}
                for index, candidate in enumerate(codes)
            ],
        }],
    }


class MultiwordRouteTests(unittest.TestCase):
    async def discover(self, message="大端 小端", *, existing=(), store=None):
        store = store or harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "s54-offline")

        async def tool(name, args, *_args, **_kwargs):
            if name == "keytao_lookup_by_word":
                return json.dumps({"success": True, "phrases": [
                    {"word": args["word"], "code": "xcdtio", "type": "Phrase"}
                ] if args["word"] in existing else []})
            if name == "keytao_pending_items_by_words":
                return json.dumps({"success": True, "complete": True, "items": []})
            if name == "keytao_prepare_reviewed_add":
                return json.dumps(review_payload(args["word"]))
            raise AssertionError(name)

        with patch.object(chat, "conversation_state_store", store), patch.object(
            chat, "call_tool_function", side_effect=tool,
        ), patch.object(chat, "_classify_simple_word_query_intent", side_effect=AssertionError("model routing")):
            response = await commands._try_handle_simple_single_word_query(message, "qq", "s54-offline", key)
        return response, store, key

    def test_bare_two_words_persists_one_reviewed_record(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s54-offline")
            calls = []

            async def tool(name, args, *_args, **_kwargs):
                calls.append((name, args))
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(review_payload(args["word"]))
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", side_effect=tool,
            ), patch.object(
                chat, "_classify_simple_word_query_intent",
                AsyncMock(return_value=harness.SimpleWordQueryIntent(
                    True, ("大端", "小端"), "word_lookup", 1.0,
                )),
            ), patch.object(store, "set", wraps=store.set) as persist:
                response = await commands._try_handle_simple_single_word_query(
                    "大端 小端", "qq", "s54-offline", key,
                )
                self.assertIsNotNone(response, "bare multiword query fell through")
                self.assertEqual(persist.call_count, 1)
                state = store.get(key)
                self.assertIsInstance(state, harness.PendingToolConfirm)
                self.assertTrue(state.args["_reviewed_multi_word"])
                self.assertEqual([s["word"] for s in state.args["_candidate_scopes"]], ["大端", "小端"])
                self.assertEqual(response.count("审词："), 2)
                self.assertIn("dà duān", response)
                self.assertIn("常用度证据不足", response)
                self.assertTrue(chat._advertised_reply_matches_live_record(response, store.get_record(key)))
                self.assertEqual(chat._enforce_advertised_reply_contract(response, key), response)
            self.assertEqual([args["word"] for name, args in calls if name == "keytao_prepare_reviewed_add"], ["大端", "小端"])
        asyncio.run(run())

    def test_numbered_candidates_without_record_are_refused(self):
        key = harness.ConversationAddress.private("qq", "s54-boundary")
        store = harness.MemoryConversationStateStore()
        for code in ("dsdt", "`dsdt`", "**dsdt**"):
            response = f"候选：\n1. {code} — 已有「打断」"
            with patch.object(chat, "conversation_state_store", store):
                delivered = chat._enforce_advertised_reply_contract(response, key)
            self.assertNotIn("dsdt", delivered)
            self.assertIn("未", delivered)

    def test_separators_three_words_and_existing_word_share_pipeline(self):
        async def run():
            for text in ("大端 小端 肌群", "大端、小端、肌群", "大端，小端，肌群", "大端\n小端\t肌群"):
                response, store, key = await self.discover(text)
                self.assertEqual(response.count("审词："), 3)
                self.assertEqual(len(store.get(key).args["items"]), 3)
            response, store, key = await self.discover(existing=("小端",))
            self.assertIn("已在词库", response)
            self.assertNotIn("换码", response)
            self.assertEqual([i["word"] for i in store.get(key).args["items"]], ["大端"])
            self.assertTrue(chat._advertised_reply_matches_live_record(response, store.get_record(key)))
            compact_reply = "\n".join(line for line in response.splitlines() if line.strip())
            self.assertTrue(chat._advertised_reply_matches_live_record(compact_reply, store.get_record(key)))
            with patch.object(chat, "conversation_state_store", store):
                self.assertEqual(chat._enforce_advertised_reply_contract(compact_reply, key), compact_reply)
        asyncio.run(run())

    def test_cap_stops_before_any_tool_and_verbs_are_not_lexical_items(self):
        async def run():
            words = ["词" + char for char in "甲乙丙丁戊己庚辛壬癸子"]
            with patch.object(chat, "call_tool_function", side_effect=AssertionError("over cap")):
                result = await commands._try_handle_simple_single_word_query(" ".join(words), "qq", "s54")
            self.assertIn("最多查询 10 个词", result)
            self.assertEqual(commands._bare_multi_word_query_words("加 小端"), ())
            self.assertEqual(commands._bare_multi_word_query_words("删除 大端"), ())
        asyncio.run(run())

    def test_sqlite_reload_partial_selection_and_actor_isolation(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                path = directory + "/pending.db"
                original = SQLiteConversationStateStore(path)
                response, _store, key = await self.discover(store=original)
                reloaded = SQLiteConversationStateStore(path)
                record = reloaded.get_record(key)
                self.assertEqual(record.nonce, original.get_record(key).nonce)
                self.assertTrue(chat._advertised_reply_matches_live_record(response, record))
                execute = AsyncMock(return_value="selected draft saved")
                with patch.object(chat, "conversation_state_store", reloaded), patch.object(
                    commands, "_execute_confirmed_tool", execute,
                ):
                    other = harness.ConversationAddress.private("qq", "other-s54")
                    self.assertIsNone(await commands.handle_pending_message_core("小端 2", "qq", "other-s54", other, allow_intent_model=False))
                    receipt = await commands.handle_pending_message_core("小端 2", "qq", "s54-offline", key, allow_intent_model=False)
                selected = execute.await_args.args[0]
                self.assertEqual([(item["word"], item["code"]) for item in selected.args["items"]], [("小端", "xcdtv")])
                self.assertIn("未选择：「大端」", receipt)
                self.assertIsNone(reloaded.get_record(key))
        asyncio.run(run())

    def test_word_set_commands_are_not_bare_lexical_queries(self):
        for message in (
            "天选打工人先不要，其他可以加，沙县小吃也不要",
            "火星词先不要，其他都加",
            "大端先不加，小端也不要，其他加入",
            "把大端加入草稿，小端也加入草稿",
            "大端这条删除，小端保留",
            "大端先不要，小端也不要",
            "批量查询 大端 小端",
        ):
            with self.subTest(message=message):
                self.assertEqual(commands._bare_multi_word_query_words(message), ())
        self.assertEqual(commands._bare_multi_word_query_words("加班 加工 加法"), ("加班", "加工", "加法"))

    def test_main_chat_stage_uses_same_record_executor_and_review_seals(self):
        async def run():
            _response, store, key = await self.discover()
            result = commands.DraftActionResult("batch submitted", success=True)
            execute = AsyncMock(return_value=result)
            context = SimpleNamespace(
                response=None, generic_intent_is_fresh_command=False, conv_key=key,
                normalized_message_text="加入并提交", platform="qq", user_id="s54-offline",
                history=[], space_key=key.space_key, owner_label="S54",
            )
            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "_perform_batch_add_to_draft_and_submit", execute,
            ):
                await chat._stage_execute_pending_state(context)
            self.assertEqual(context.response, "batch submitted")
            self.assertEqual(execute.await_count, 1)
            self.assertEqual(set(execute.await_args.kwargs["reviewed_capabilities"]), {("大端", "dsdtvo"), ("小端", "xcdtvo")})
            self.assertIsNone(store.get_record(key))
        asyncio.run(run())

    def test_completed_batch_receipt_does_not_regenerate_add_suggestions(self):
        async def run():
            receipt = (
                "✅ 批次已提交审核。\n批次地址：http://localhost:3100/batch/s54\n\n"
                "- 「大端」→ dsdtvo\n- 「小端」→ xcdti"
            )
            key = harness.ConversationAddress.private("qq", "s54-receipt")
            ctx = SimpleNamespace(response=receipt, conv_key=key)
            with patch.object(chat, "conversation_state_store", harness.MemoryConversationStateStore()):
                await chat._stage_normalize_response(ctx)
                self.assertEqual(ctx.response, receipt)
                self.assertEqual(chat._enforce_advertised_reply_contract(ctx.response, key), receipt)
        asyncio.run(run())

    def test_main_stage_warning_replay_keeps_saved_reading_seals(self):
        async def run():
            _response, store, key = await self.discover()
            original = store.get(key)
            capabilities = commands._reviewed_multi_word_capabilities(original)
            warning = server_warning_pending_state(harness.PendingToolConfirm(
                function_name="keytao_batch_add_to_draft",
                args={"items": original.args["items"], "_submit_after": True,
                      "_reviewed_batch_readings": [
                          {"word": word, "pinyin": value["pinyin"], "candidate_codes": list(value["candidate_codes"])}
                          for (word, _code), value in capabilities.items()
                      ]},
            ), {"requiresConfirmation": True, "batchId": "s54-warning", "contentVersion": 1,
                "warningDigest": "a" * 64, "warnings": ["new risk"]})
            store.set(key, warning)
            execute = AsyncMock(return_value=commands.DraftActionResult("submitted", success=True))
            context = SimpleNamespace(
                response=None, generic_intent_is_fresh_command=False, conv_key=key,
                normalized_message_text="加入并提交", platform="qq", user_id="s54-offline",
                history=[], space_key=key.space_key, owner_label="S54",
            )
            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "_perform_batch_add_to_draft_and_submit", execute,
            ):
                await chat._stage_execute_pending_state(context)
            self.assertEqual(context.response, "词：大端、小端\nsubmitted")
            self.assertEqual(execute.await_args.kwargs["reviewed_capabilities"], capabilities)
            self.assertTrue(execute.await_args.kwargs["confirmed_add"])
        asyncio.run(run())

    def test_new_server_warning_replaces_selector_and_cannot_change_selection(self):
        async def run():
            original, store, key = await self.discover()
            warning = server_warning_pending_state(store.get(key), {
                "requiresConfirmation": True, "batchId": "s54-warning",
                "contentVersion": 1, "warningDigest": "a" * 64,
                "warnings": ["a newly observed occupied slot"],
            })
            store.set(key, warning)
            prompt = chat._chat_render._format_server_bound_confirmation_prompt(warning)
            with patch.object(chat, "conversation_state_store", store):
                delivered = chat._enforce_advertised_reply_contract(prompt, key)
                self.assertEqual(delivered, prompt)
                self.assertIn("a newly observed occupied slot", delivered)
                self.assertNotIn("大端 3", delivered)
                selected, intent, refusal = harness.chat_routing_module._resolve_multi_word_pending_candidate_selection(warning, "大端 2")
                self.assertIsNone(selected)
                self.assertIsNone(intent)
                self.assertIn("风险确认", refusal)
                self.assertFalse(chat._advertised_reply_matches_live_record(original, store.get_record(key)))
        asyncio.run(run())

    def test_same_codes_with_wrong_words_redraw_from_record(self):
        async def run():
            response, store, key = await self.discover()
            forged = "\n".join(line for line in response.replace("大端", "错误词甲").replace("小端", "错误词乙").splitlines() if not line.startswith("- "))
            with patch.object(chat, "conversation_state_store", store):
                delivered = chat._enforce_advertised_reply_contract(forged, key)
            self.assertEqual(delivered, response)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
