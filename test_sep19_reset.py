"""Offline compound reset authorization and stale-state regressions."""

import asyncio
import copy
import json
import socket
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import test_state_machine as harness

chat = harness.openai_chat_module
routing = chat._chat_routing
REQUEST = "清空草稿和缓存，忘记我对你说过的话"
REAL_CLEAR_CURRENT_DRAFT = chat._perform_clear_current_draft


class CompoundResetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.store = harness.MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="reset-owner", space_type="group", space_id="reset-room",
        )
        self.key = self.memory.conversation_address
        self.other = harness.ConversationAddress.group("qq", "reset-room", "other-owner")
        self.other_room = harness.ConversationAddress.group("qq", "other-room", "reset-owner")
        for key in (self.key, self.other, self.other_room):
            self.store.set(key, harness.PendingAddWord(
                word="旧词", recommended_code="abcd", candidates=[("abcd", False)],
            ))
        self.coordinator = harness.DraftOperationCoordinator()
        self.memory_clear = Mock()
        self.history_clear = Mock()
        self.perform_clear = AsyncMock(return_value=harness.DraftActionResult(
            "✅ 已清空草稿，共删除 1 条。", success=True,
        ))
        self.delivered = AsyncMock()
        self.tasks = {}
        for patcher in (
            patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")),
            patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")),
            patch.object(chat, "conversation_state_store", self.store),
            patch.object(chat, "draft_operation_coordinator", self.coordinator),
            patch.object(chat, "memory_store", SimpleNamespace(clear_conversation=self.memory_clear)),
            patch.object(chat, "clear_history", self.history_clear),
            patch.object(chat, "memory_compaction_tasks", {}),
            patch.object(chat, "background_draft_tasks_by_conversation", self.tasks),
            patch.object(chat, "_perform_clear_current_draft", self.perform_clear),
            patch.object(chat, "_finish_ai_chat_response", self.delivered),
            patch.object(chat, "_finish_ai_chat_matcher", self.delivered),
            patch.object(chat, "get_history", return_value=[]),
            patch.object(chat, "remember_conversation", side_effect=AssertionError("Reset must not recreate history")),
            patch.object(chat, "call_tool_function", side_effect=AssertionError("Unexpected external tool")),
            patch.object(chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("Old pending reached a model before reset"))),
            patch.object(chat, "get_ai_response_core", AsyncMock(side_effect=AssertionError("Model forbidden"))),
        ):
            self.stack.enter_context(patcher)

    def context(self, message=REQUEST):
        cache, classifier = chat.command_intent_memoizer(message, message)
        return chat.TurnContext(
            bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
            message_text=message, normalized_message_text=message,
            conv_key=self.key, space_key=self.key.space_key, memory_context=self.memory,
            command_intent_cache=cache, command_intent_for=classifier,
            reply_reference=harness.ReplyReferenceInfo(),
        )

    async def run_early_stages(self, ctx):
        start = chat.STAGES.index(chat._stage_initialize_conversation) + 1
        end = chat.STAGES.index(chat._stage_generate_ai_response)
        for stage in chat.STAGES[start:end + 1]:
            if await stage(ctx):
                return True
        return False

    async def test_reset_preempts_stale_pending_without_models(self):
        ctx = self.context()
        self.assertTrue(await self.run_early_stages(ctx))
        self.perform_clear.assert_awaited_once_with("qq", "reset-owner")
        self.memory_clear.assert_called_once_with(self.memory)
        self.history_clear.assert_called_once_with(self.key)
        self.assertIsNone(self.store.get(self.key))
        self.assertIsNone(self.store.get(self.other_room))
        self.assertIsNotNone(self.store.get(self.other))
        self.assertIn("已清空草稿", ctx.response)
        self.assertIn("对话", ctx.response)
        self.assertNotIn("旧词", ctx.response)
        self.assertEqual(self.delivered.await_count, 1)

    async def test_draft_failure_still_clears_only_current_conversation(self):
        self.perform_clear.return_value = harness.DraftActionResult("获取草稿失败：fixture unavailable")
        ctx = self.context()
        self.assertTrue(await self.run_early_stages(ctx))
        self.assertIn("获取草稿失败", ctx.response)
        self.assertNotIn("已清空草稿", ctx.response)
        self.memory_clear.assert_called_once_with(self.memory)
        self.history_clear.assert_called_once_with(self.key)
        self.assertIsNotNone(self.store.get(self.other_room))
        self.assertIsNotNone(self.store.get(self.other))

    async def test_running_write_is_preserved_and_not_claimed_cleared(self):
        operation = self.coordinator.begin(self.key, "add", word="正在写入", code="abcd")
        self.coordinator.mark_running(self.key, operation.operation_id)
        wait = asyncio.Event()
        task = asyncio.create_task(wait.wait())
        self.tasks[self.key] = {task}
        try:
            ctx = self.context()
            self.assertTrue(await self.run_early_stages(ctx))
            self.perform_clear.assert_not_awaited()
            self.assertFalse(task.cancelled())
            self.assertIs(self.coordinator.get(self.key), operation)
            self.assertNotIn("已清空草稿", ctx.response)
            self.assertIn("进行中", ctx.response)
            self.memory_clear.assert_called_once_with(self.memory)
        finally:
            wait.set()
            await task

    async def test_awaiting_operation_is_cancelled_before_clearing_draft(self):
        operation = self.coordinator.begin(self.key, "add", word="旧词", code="abcd")
        self.coordinator.mark_awaiting_confirmation(
            self.key, operation.operation_id,
            harness.PendingToolConfirm(function_name="keytao_create_phrase", args={}),
            "旧确认提示",
        )
        ctx = self.context()
        self.assertTrue(await self.run_early_stages(ctx))
        self.perform_clear.assert_awaited_once_with("qq", "reset-owner")
        self.assertIsNone(self.coordinator.get(self.key))
        self.assertNotIn("旧词", ctx.response)
        self.assertNotIn("正等待", ctx.response)
        self.assertIn("已清空草稿", ctx.response)

    async def test_unexpected_draft_failure_does_not_claim_deletion(self):
        self.perform_clear.side_effect = RuntimeError("Offline failure")
        ctx = self.context()
        self.assertTrue(await self.run_early_stages(ctx))
        self.assertNotIn("已清空草稿", ctx.response)
        self.assertIn("未能确认", ctx.response)
        self.memory_clear.assert_called_once_with(self.memory)

    async def test_conversation_failure_does_not_claim_forgetting_succeeded(self):
        self.memory_clear.side_effect = RuntimeError("Offline store failure")
        ctx = self.context()
        self.assertTrue(await self.run_early_stages(ctx))
        self.assertIn("已清空草稿", ctx.response)
        self.assertIn("对话清理未全部完成", ctx.response)
        self.assertNotIn("记忆和待确认状态已清空", ctx.response)

    async def test_real_clear_binds_actor_batch_version_and_exact_targets(self):
        for drift in (False, True):
            with self.subTest(drift=drift):
                calls = []
                items = [{"id": 19, "word": "旧词", "code": "abcd",
                          "action": "Create", "type": "Phrase"}]

                async def tool(name, args, platform, user_id):
                    self.assertEqual((platform, user_id), ("qq", "reset-owner"))
                    calls.append((name, copy.deepcopy(args)))
                    if name == "keytao_list_draft_items":
                        self.assertIn(args, ({}, {"batch_id": "own-draft"}))
                        return json.dumps({"success": True, "batchId": "own-draft",
                                           "contentVersion": 7, "items": items})
                    self.assertEqual(name, "keytao_batch_remove_draft_items")
                    self.assertEqual(args["ids"], [19])
                    self.assertEqual(args["batch_id"], "own-draft")
                    if not args.get("expected_target_digest"):
                        return json.dumps({
                            "success": False, "requiresConfirmation": True,
                            "batchId": "own-draft", "contentVersion": 8 if drift else 7,
                            "targetDigest": "d" * 64, "targets": items,
                        })
                    self.assertFalse(drift, "Drifted preview must never reach the write")
                    self.assertEqual(args["expected_content_version"], 7)
                    self.assertEqual(args["expected_target_digest"], "d" * 64)
                    self.assertEqual(args["expected_targets"], items)
                    items.clear()
                    return json.dumps({"success": True, "batchId": "own-draft"})

                with patch.object(chat, "_perform_clear_current_draft", REAL_CLEAR_CURRENT_DRAFT), patch.object(
                    chat, "call_tool_function", side_effect=tool,
                ):
                    ctx = self.context()
                    self.assertTrue(await self.run_early_stages(ctx))
                mutations = [args for name, args in calls if name == "keytao_batch_remove_draft_items"]
                self.assertEqual(len(mutations), 1 if drift else 2)
                if drift:
                    self.assertIn("发生变化", ctx.response)
                    self.assertNotIn("已清空草稿", ctx.response)
                else:
                    self.assertEqual(items, [])
                    self.assertIn("已清空草稿", ctx.response)

    async def test_non_commands_cannot_reach_either_clear(self):
        stage = getattr(chat, "_stage_handle_explicit_draft_conversation_reset", None)
        self.assertIsNotNone(stage)
        denied = [
            "不要" + REQUEST, REQUEST + "吗", REQUEST + "？", "他说" + REQUEST,
            "请解释" + REQUEST, "如果" + REQUEST, "「" + REQUEST + "」",
            "“" + REQUEST + "”", "清空别人的草稿和缓存，忘记我对你说过的话",
            REQUEST + "，再提交草稿", "加词新词，" + REQUEST,
            "清空草稿和全局缓存，忘记我对你说过的话",
            REQUEST + "\n", REQUEST.replace("缓存", "缓\x00存"),
        ]
        for message in denied:
            with self.subTest(message=message):
                ctx = self.context(message)
                ctx.normalized_message_text = REQUEST
                self.assertFalse(await stage(ctx))
        self.perform_clear.assert_not_awaited()
        self.memory_clear.assert_not_called()
        self.history_clear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
