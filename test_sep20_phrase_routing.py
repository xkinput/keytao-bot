"""Offline literal routing through the real early chat stage sequence."""

import copy
import json
import socket
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import test_state_machine as harness


chat = harness.openai_chat_module
commands = harness.chat_commands_module
PHRASE = "头痛医头，脚痛医脚"
MESSAGES = ("“" + PHRASE + "”", "将“" + PHRASE + "”作为单个词加入词库")
STATE_KINDS = ("None", "PendingAddWord", "PendingToolConfirm", "PendingTrustedWordRecord")


class LiteralPhraseEarlyRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def replay(self, message, kind="None", *, allow_classifier=False,
                     history=None, active_status=None):
        store = harness.MemoryConversationStateStore()
        memory = harness.ChatMemoryContext(
            platform="qq", user_id="literal-review", space_type="private",
        )
        key = memory.conversation_address
        if kind == "PendingAddWord":
            state = harness.PendingAddWord(
                word="旧词", recommended_code="abcd", candidates=[("abcd", False)],
                server_candidates=[("abcd", False)],
            )
        elif kind == "PendingToolConfirm":
            state = harness.PendingToolConfirm("keytao_batch_add_to_draft", {
                "items": [{"word": "旧词", "code": "abcd", "type": "Phrase", "action": "Create"}],
            })
        elif kind == "PendingTrustedWordRecord":
            state = commands.create_pending_trusted_word_record("旧词", "abcd", "Phrase")
        else:
            state = None
        if state is not None:
            self.assertTrue(store.set(key, state))
        coordinator = harness.DraftOperationCoordinator()
        operation = None
        if active_status is not None:
            operation = coordinator.begin(key, "add", word="执行中旧词", code="abcd")
            self.assertIsNotNone(operation)
            if active_status == "running":
                self.assertTrue(coordinator.mark_running(key, operation.operation_id))
            elif active_status == "awaiting_confirmation":
                self.assertTrue(coordinator.mark_awaiting_confirmation(
                    key, operation.operation_id,
                    harness.PendingToolConfirm("keytao_create_phrase", {
                        "word": "执行中旧词", "code": "abcd", "phrase_type": "Phrase",
                    }),
                    "旧任务等待确认",
                ))
        operation_before = copy.deepcopy(operation)
        ctx = chat.TurnContext(
            bot=object(), event=object(), platform="qq", user_id=key.actor_id,
            message_text=message, reply_reference=harness.ReplyReferenceInfo(),
        )
        calls, stages, classifications, delivered = [], [], [], []

        async def classifier(_text, pending=None):
            classifications.append((stages[-1], type(pending).__name__))
            if not allow_classifier:
                raise AssertionError("Intent model reached before deterministic lookup")
            return harness.MessageCommandIntent()

        async def tool(name, args, platform, user_id):
            self.assertEqual((platform, user_id), ("qq", key.actor_id))
            calls.append((name, copy.deepcopy(args)))
            if name == "keytao_lookup_by_word":
                return json.dumps({"success": True, "phrases": []})
            if name == "keytao_pending_items_by_words":
                return json.dumps({"success": True, "complete": True, "items": []})
            if name == "keytao_prepare_reviewed_add":
                return json.dumps({"success": False, "message": "Fixture exact literal review reached"})
            raise AssertionError("Unexpected external tool: " + name)

        async def finish(*args, **_kwargs):
            delivered.append(str(args[-2] if len(args) >= 5 else args[0]))

        word_classifier = AsyncMock(return_value=harness.SimpleWordQueryIntent(False))
        main_model = AsyncMock(side_effect=AssertionError("Main model forbidden"))
        schedule = AsyncMock(side_effect=AssertionError("Background mutation forbidden"))
        with ExitStack() as stack:
            for patcher in (
                patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
                patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")),
                patch.object(socket, "create_connection", side_effect=AssertionError("Network forbidden")),
                patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")),
                patch.object(chat, "conversation_state_store", store),
                patch.object(chat, "draft_operation_coordinator", coordinator),
                patch.object(chat, "extract_memory_context", AsyncMock(return_value=memory)),
                patch.object(chat, "get_history", return_value=history or []),
                patch.object(chat, "remember_conversation"),
                patch.object(chat, "schedule_memory_compaction"),
                patch.object(chat, "_classify_message_command_intent", AsyncMock(side_effect=classifier)),
                patch.object(chat, "_classify_simple_word_query_intent", word_classifier),
                patch.object(chat, "get_ai_response_core", main_model),
                patch.object(chat, "_schedule_background_draft_operation", schedule),
                patch.object(chat, "call_tool_function", side_effect=tool),
                patch.object(chat, "_finish_ai_chat_response", side_effect=finish),
                patch.object(chat, "_finish_ai_chat_matcher", side_effect=finish),
            ):
                stack.enter_context(patcher)
            start = chat.STAGES.index(chat._stage_normalize_message_text)
            end = chat.STAGES.index(chat._stage_handle_simple_word_query)
            for stage in chat.STAGES[start:end + 1]:
                stages.append(stage.__name__)
                if await stage(ctx):
                    break
        return {
            "ctx": ctx, "calls": calls, "stages": stages, "classifications": classifications,
            "delivered": delivered, "word_classifier": word_classifier, "main_model": main_model,
            "schedule": schedule, "operation": coordinator.get(key), "operation_before": operation_before,
        }

    def assert_exact_literal_lookup(self, result):
        self.assertEqual(result["stages"][-1], "_stage_handle_simple_word_query", result["delivered"])
        self.assertEqual(result["classifications"], [])
        result["word_classifier"].assert_not_awaited()
        result["main_model"].assert_not_awaited()
        result["schedule"].assert_not_called()
        self.assertEqual(
            [args["word"] for name, args in result["calls"] if name == "keytao_lookup_by_word"],
            [PHRASE],
        )
        self.assertEqual(
            [args["word"] for name, args in result["calls"] if name == "keytao_prepare_reviewed_add"],
            [PHRASE],
        )
        self.assertEqual(result["ctx"].response, "Fixture exact literal review reached")

    async def test_original_sixteen_case_early_stage_matrix(self):
        for allow_classifier in (False, True):
            for kind in STATE_KINDS:
                for message in MESSAGES:
                    with self.subTest(allow_classifier=allow_classifier, state=kind, message=message):
                        result = await self.replay(message, kind, allow_classifier=allow_classifier)
                        self.assert_exact_literal_lookup(result)

    async def test_old_history_advertisement_does_not_block_new_literal(self):
        histories = (
            [{"role": "assistant", "content": "候选词：旧词 abcd。回复“加入”即可添加。"}],
            [{"role": "assistant", "content": "旧词可添加到草稿，你想继续吗？"}],
        )
        for history in histories:
            for message in MESSAGES:
                with self.subTest(history=history, message=message):
                    self.assert_exact_literal_lookup(await self.replay(message, history=history))

    async def test_active_operations_keep_their_identity_status_and_confirmation(self):
        for status in ("queued", "awaiting_confirmation", "running"):
            for kind in ("None", "PendingToolConfirm"):
                for message in MESSAGES:
                    with self.subTest(status=status, state=kind, message=message):
                        result = await self.replay(message, kind, active_status=status)
                        self.assert_exact_literal_lookup(result)
                        self.assertEqual(result["operation"], result["operation_before"])
                        self.assertIn("_stage_arbitrate_active_operation", result["stages"])

    async def test_reported_negated_and_extra_action_messages_do_not_get_literal_bypass(self):
        command = MESSAGES[1]
        for message in (
            "他说" + MESSAGES[0], "转发：" + command, "不要" + command,
            command + "吗", command + "？", command + "并提交",
            "“" + command + "”", "“头痛医头，提交草稿”",
        ):
            with self.subTest(message=message):
                result = await self.replay(message, "PendingAddWord", allow_classifier=True)
                self.assertFalse(getattr(result["ctx"], "literal_phrase_word", ""))
                self.assertEqual(result["calls"], [])
                result["schedule"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
