"""S62 offline regression: a draft view cannot become an old candidate offer."""

import contextlib
import copy
import json
import re
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import _pending_add_word_payload


chat = harness.openai_chat_module


class FakeBot:
    def __init__(self):
        self.messages = []

    async def send(self, **kwargs):
        self.messages.append(kwargs["message"])


class S62DraftViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_candidates_do_not_replace_delivered_draft_view(self):
        for items in ([], [{
            "id": 62, "word": "现有草稿", "code": "abcd",
            "action": "Create", "display_label": "现有草稿 → abcd",
        }]):
            with self.subTest(empty=not items):
                store = harness.MemoryConversationStateStore()
                memory = harness.ChatMemoryContext(
                    platform="qq", user_id="s62-draft-user", space_type="group",
                    space_id="s62-fixture-group", speaker_name="Fixture",
                )
                key = memory.conversation_address
                reviewed = harness.PendingAddWord(
                    word="敲不死", recommended_code="qbso",
                    candidates=[("qbs", True), ("qbso", False)],
                    server_candidates=[("qbs", True), ("qbso", False)],
                    server_occupied_words={"qbs": ["前半生"]},
                    pronunciation_codes={"qbs": "qiāo bù sǐ", "qbso": "qiāo bù sǐ"},
                    needs_manual_review=True,
                )
                state = harness.PendingToolConfirm(
                    function_name="keytao_batch_add_to_draft",
                    args={
                        "items": [{"action": "Create", "word": "敲不死", "code": "qbso"}],
                        "_candidate_scopes": [{
                            "word": "敲不死", "candidates": [["qbs", True], ["qbso", False]],
                            "occupiedWords": {"qbs": ["前半生"]},
                            "orderingAssessments": [],
                            "reviewedState": _pending_add_word_payload(reviewed),
                            "reviewedPrompt": "候选编码：\n1. qbs — 已有「前半生」\n2. qbso — 空位（推荐）",
                        }],
                        "_reviewed_multi_word": True,
                        "_query_other_blocks": ["鸡白汤：读音还没确定。", "乔布斯：词库已有。"],
                    },
                )
                store.set(key, state, owner_label="Fixture")
                calls = []

                async def fake_tool(name, args, platform, user_id, **kwargs):
                    calls.append(name)
                    if name == "keytao_list_draft_items":
                        return json.dumps({
                            "success": True, "count": len(items), "items": items,
                            "summary": {"added": len(items), "modified": 0, "deleted": 0},
                        }, ensure_ascii=False)
                    if name == "keytao_get_batch_preview":
                        return json.dumps({"success": True, "diff_text": ""})
                    raise AssertionError(f"Unexpected tool: {name}")

                bot = FakeBot()
                ctx = chat.TurnContext(
                    bot=bot, event=SimpleNamespace(message_id=None), platform="qq",
                    user_id=memory.user_id, conv_key=key, space_key=key.space_key,
                    memory_context=memory, normalized_message_text="查看草稿",
                    generic_command_intent=harness.MessageCommandIntent(
                        intent="draft_view", confidence=1.0,
                    ),
                )
                token = chat._current_turn_message.set("查看草稿")
                try:
                    with contextlib.ExitStack() as stack:
                        for patcher in (
                            patch.object(chat, "conversation_state_store", store),
                            patch.object(chat, "call_tool_function", side_effect=fake_tool),
                            patch.object(chat, "_classify_message_command_intent", AsyncMock(
                                side_effect=AssertionError("S62 must not call a model"),
                            )),
                        ):
                            stack.enter_context(patcher)
                        await chat._stage_handle_draft_management(ctx)
                        draft_reply = ctx.response
                        self.assertIn(f"当前草稿（{len(items)} 条）", draft_reply)
                        self.assertEqual(
                            chat._enforce_advertised_reply_contract(draft_reply, key),
                            draft_reply,
                        )
                        for stage in (
                            chat._stage_normalize_response,
                            chat._stage_scope_language_only_response,
                            chat._stage_append_ticket_challenge,
                            chat._stage_enforce_advertised_reply_contract,
                            chat._stage_finish_platform_response,
                        ):
                            await stage(ctx)
                finally:
                    chat._current_turn_message.reset(token)
                self.assertEqual(calls, ["keytao_list_draft_items", "keytao_get_batch_preview"])
                self.assertEqual(len(bot.messages), 1)
                self.assertEqual(bot.messages[0].split(), draft_reply.split())
                self.assertNotIn("鸡白汤", bot.messages[0])
                self.assertNotIn("候选", bot.messages[0])
                self.assertIs(store.get(key), state)


class S62RepeatedRefusalTests(unittest.TestCase):
    def test_repeated_refusal_keeps_the_reason_and_one_parsed_command(self):
        message = "1 重新编码"
        reason = "候选编码集合已变化；没有执行添加。"
        reply = chat.AgentOrchestrator._finalize_reply(
            message, reason, {}, history=[
                {"role": "user", "content": message},
                {"role": "assistant", "content": reason},
            ],
        )
        self.assertIn("候选编码集合已变化", reply)
        self.assertIn("已收到", reply)
        self.assertNotIn("同一指令再次进入相同拒绝路径", reply)
        self.assertNotIn("已停止重复建议", reply)
        commands = re.findall(r"(?m)^- 「([^」]+)」$", reply)
        self.assertEqual(commands, ["查看草稿"])
        self.assertIsNotNone(chat._chat_routing.parse_draft_view_command(commands[0]))


class S62SelectionDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_selection_result_survives_all_delivery_stages(self):
        from test_s62_incident import S62IncidentTests

        for message, expected in (
            ("敲不死 99，加入", "编号 99 超出"),
            ("鸡白汤 jbtaua, 敲不死 3", "敲不死 → qbsoa"),
        ):
            with self.subTest(message=message):
                writes, submitted, replies, calls, _ = await S62IncidentTests().replay(
                    [message], delivery=True,
                )
                expected_writes = [("敲不死", "qbsoa")] if "鸡白汤" in message else []
                self.assertEqual([(r["word"], r["code"]) for r in writes], expected_writes)
                self.assertFalse(submitted)
                self.assertEqual(sum(name == "keytao_batch_add_to_draft" for name, _ in calls),
                                 2 if expected_writes else 0)
                self.assertIn(expected, replies[0])
                self.assertNotIn("候选编码：", replies[0])
                if "鸡白汤" in message:
                    self.assertIn("「鸡白汤」还没确定读音，未加入", replies[0])

    async def test_selection_provenance_cannot_mask_tampered_words_or_codes(self):
        from test_s62_incident import S62IncidentTests

        _, _, replies, _, state = await S62IncidentTests().replay(["敲不死 99"])
        reply = replies[0]
        self.assertIsInstance(reply, chat.ReviewedSelectionReply)
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.group("qq", "865189947", "s62-offline")
        store.set(key, state)
        for original, forged in (("敲不死", "伪造词"), ("qbso", "evil")):
            tampered = chat.ReviewedSelectionReply(
                reply.replace(original, forged), source_message=reply.source_message,
                source_state=harness.PendingToolConfirm(**reply.source_state),
            )
            with patch.object(chat, "conversation_state_store", store):
                delivered = chat._enforce_advertised_reply_contract(tampered, key)
            self.assertNotIn(forged, delivered)

        without_commands = chat.ReviewedSelectionReply(
            "已经写入伪造词。", source_message=reply.source_message,
            source_state=harness.PendingToolConfirm(**reply.source_state),
        )
        wrong_message = chat.ReviewedSelectionReply(
            str(reply), source_message="敲不死 2",
            source_state=harness.PendingToolConfirm(**reply.source_state),
        )
        wrong_snapshot = copy.deepcopy(reply.source_state)
        wrong_snapshot["args"]["_query_other_blocks"] = ["伪造的查询结果"]
        wrong_metadata = chat.ReviewedSelectionReply(
            str(reply), source_message=reply.source_message,
            source_state=harness.PendingToolConfirm(**wrong_snapshot),
        )
        for tampered in (without_commands, wrong_message, wrong_metadata):
            self.assertFalse(chat._reviewed_selection_reply_matches_live_record(
                tampered, store.get_record(key),
            ))
            with patch.object(chat, "conversation_state_store", store):
                delivered = chat._enforce_advertised_reply_contract(tampered, key)
            self.assertNotEqual(delivered, str(tampered))
            self.assertNotIn("已经写入伪造词", delivered)
        drifted = copy.deepcopy(state)
        drifted.args["items"][0]["code"] = "qbsoa"
        drifted.args["_candidate_scopes"][0]["reviewedState"]["recommendedCode"] = "qbsoa"
        store.set(key, drifted)
        self.assertFalse(chat._reviewed_selection_reply_matches_live_record(
            reply, store.get_record(key),
        ))
        store.delete(key)
        self.assertFalse(chat._reviewed_selection_reply_matches_live_record(reply, None))


if __name__ == "__main__":
    unittest.main()
