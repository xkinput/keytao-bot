"""B19 remainder replies retain send-backed QQ ticket bindings offline."""

import copy
import json
import unittest
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import urlsplit

import test_b18_rest_real_tools as b18


harness = b18.harness
chat = harness.openai_chat_module
REAL_FINISH = chat._finish_ai_chat_response
REAL_MATCHER = chat._finish_ai_chat_matcher
WORDS = ("马哲", *(word for word in b18.TEN_WORDS if word != "宜居"))
REST = WORDS[1:]
REFUSAL = "这条回复未对应当前待确认操作；请引用当前候选消息后再回复。"


class OfflineQQBot:
    __module__ = "nonebot.adapters.onebot.v11"

    def __init__(self):
        self.self_id = "3785773770"
        self.messages = {}
        self.next_id = 1900
        self.send = AsyncMock(side_effect=self.deliver)
        self.get_msg = AsyncMock(side_effect=lambda message_id: copy.deepcopy(self.messages[str(message_id)]))

    async def deliver(self, *, event, message):
        message_id = str(self.next_id)
        self.next_id += 1
        self.messages[message_id] = {
            "sender": {"user_id": self.self_id}, "message": str(message),
        }
        self.last_id = message_id
        return {"message_id": int(message_id)}


class DraftHTTP(b18.DraftHTTP):
    def _response(self, method, url, kwargs):
        if method == "GET" and urlsplit(url).path == f"/api/batches/{self.batch_id}":
            self.requests.append((method, urlsplit(url).path, {}))
            return b18.OfflineResponse({"batch": {
                "id": self.batch_id, "status": "Draft", "creatorId": 42,
                "contentVersion": self.version,
                "pullRequests": [{**row, "userId": 42} for row in self.rows],
            }})
        return super()._response(method, url, kwargs)


class B19PartialReplyBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        b18.B18RealToolTests.setUp(self)
        self.http = DraftHTTP()
        self.enterContext(patch.object(b18.TOOLS.http_client, "_keytao_client", self.http))
        self.bot = OfflineQQBot()
        module = ModuleType("nonebot.adapters.onebot.v11")
        module.Bot = OfflineQQBot
        self.enterContext(patch.dict("sys.modules", {module.__name__: module}))
        self.enterContext(patch.object(chat, "_finish_ai_chat_response", REAL_FINISH))
        self.enterContext(patch.object(chat, "_finish_ai_chat_matcher", REAL_MATCHER))
        self.enterContext(patch.object(chat.ai_chat, "send", AsyncMock(side_effect=self.matcher_send), create=True))
        self.enterContext(patch.object(chat.ai_chat, "finish", AsyncMock(side_effect=self.matcher_finish)))
        self.info = self.enterContext(patch.object(chat.logger, "info", wraps=chat.logger.info))

    async def matcher_send(self, response):
        return await self.bot.send(event=None, message=response)

    async def matcher_finish(self, response=None):
        if response is not None:
            return await self.matcher_send(response)

    def tearDown(self):
        b18.B18RealToolTests.tearDown(self)

    def created(self):
        return b18.B18RealToolTests.created(self)

    async def seed(self, words=WORDS):
        record = b18.B18RealToolTests.seed(self, words, reorder=False)
        await REAL_FINISH(self.bot, SimpleNamespace(message_id=1), self.replay.author.actor_id,
                          self.replay.memory, self.replay.prompt)
        self.original_id = self.bot.last_id
        self.assertEqual(record.origin_message_ids, (self.original_id,))
        self.assertTrue(any("branch=send_backed" in str(call.args[0]) for call in self.info.call_args_list))
        return record

    async def turn(self, message, reply_id, actor=None, raw=None):
        actor = actor or self.replay.author.actor_id
        memory = replace(self.replay.memory, user_id=actor)
        full_message = f"[reply:id={reply_id}][at:qq={self.bot.self_id}] {message}"
        event = SimpleNamespace(
            to_me=True, group_id=self.replay.author.space_id, user_id=actor, message_id=2,
            raw_message=full_message,
            message=[SimpleNamespace(type="reply", data={"id": reply_id}),
                     SimpleNamespace(type="at", data={"qq": self.bot.self_id}),
                     SimpleNamespace(type="text", data={"text": message})],
            get_plaintext=lambda: message,
        )
        tokens = [(variable, variable.set(value)) for variable, value in (
            (chat._current_turn_message, full_message),
            (chat._current_conversational_turn, None),
            (chat.current_operation_capture, None),
            (chat.current_draft_delivery_claims, []),
        )]
        self.replay.client.create.return_value.choices[0].message.content = json.dumps(
            raw or b18.raw_operation(op="cancel" if message == "取消" else "confirm" if message in (
                "加入", "加入并提交", "其它的用默认编码",
            ) else "select_candidate", refers_to_pending=True, operands=[], codes=[]), ensure_ascii=False,
        )
        before = self.bot.send.await_count
        try:
            with patch.object(chat, "extract_memory_context", AsyncMock(return_value=memory)), patch.object(
                chat, "_classify_message_command_intent", AsyncMock(return_value=chat.MessageCommandIntent()),
            ):
                await chat._handle_ai_chat_serialized(self.bot, event, "qq", actor)
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)
        self.assertEqual(self.bot.send.await_count, before + 1)
        self.bot.get_msg.assert_awaited_with(message_id=int(reply_id))
        self.replay.last_response = self.bot.messages[self.bot.last_id]["message"]
        return self.bot.last_id

    async def partial(self):
        original = await self.seed()
        self.info.reset_mock()
        partial_id = await self.turn("马哲 2 5", self.original_id)
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")], self.replay.last_response)
        remaining = self.replay.store.get_record(self.replay.author)
        self.assertEqual(remaining.state.args["_query_words"], list(REST))
        self.assertIn(self.original_id, remaining.origin_message_ids)
        self.assertEqual((remaining.created_at, remaining.expires_at), (original.created_at, original.expires_at))
        self.assertIn("未选择：", self.replay.last_response)
        return partial_id

    async def complete(self, message, quote_partial):
        partial_id = await self.partial()
        await self.turn(message, partial_id if quote_partial else self.original_id)
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")] + [
            (word, b18.FACTS[word][0]) for word in REST
        ], self.replay.last_response)
        self.assertEqual(self.http.submissions, int("提交" in message), self.replay.last_response)
        pending = self.replay.store.get_record(self.replay.author)
        if "提交" in message:
            self.assertIsNone(pending, self.replay.last_response)
        else:
            self.assertTrue(pending is None or pending.state.function_name == "keytao_submit_batch", self.replay.last_response)

    async def test_r1_partial_reply_default_writes_remaining(self):
        await self.complete("其它的用默认编码", True)

    async def test_r1_partial_reply_add_writes_remaining(self):
        await self.complete("加入", True)

    async def test_r1_partial_reply_add_submit_writes_remaining(self):
        await self.complete("加入并提交", True)

    async def test_r1_original_reply_default_writes_remaining(self):
        await self.complete("其它的用默认编码", False)

    async def test_r1_original_reply_add_writes_remaining(self):
        await self.complete("加入", False)

    async def test_r1_original_reply_add_submit_writes_remaining(self):
        await self.complete("加入并提交", False)

    async def test_r1_partial_reply_send_binds_actual_message_id_and_digest(self):
        partial_id = await self.partial()
        record = self.replay.store.get_record(self.replay.author)
        self.assertIn(partial_id, record.origin_message_ids)
        self.assertEqual(record.origin_prompt_digest, chat._prompt_capability_digest(self.replay.last_response))
        self.assertTrue(self.replay.store.reply_message_id_matches(record, partial_id))
        self.assertTrue(any("branch=send_backed" in str(call.args[0]) for call in self.info.call_args_list))

    async def test_r1_partial_reply_cancel_preserves_written_rows(self):
        partial_id = await self.partial()
        writes = copy.deepcopy(self.http.writes)
        await self.turn("取消", partial_id)
        self.assertEqual(self.replay.last_response, "已取消。")
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assertEqual(self.http.writes, writes)

    async def test_r1_original_reply_cancel_preserves_written_rows(self):
        await self.partial()
        writes = copy.deepcopy(self.http.writes)
        await self.turn("取消", self.original_id)
        self.assertEqual(self.replay.last_response, "已取消。")
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assertEqual(self.http.writes, writes)

    async def test_r3_later_partial_reply_binds_smaller_remainder_without_written_words(self):
        partial_id = await self.partial()
        smaller_id = await self.turn("小美 2", partial_id)
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea"), ("小美", "xmwwi")],
                              self.replay.last_response)
        remaining = self.replay.store.get_record(self.replay.author)
        smaller = [word for word in REST if word != "小美"]
        self.assertEqual(remaining.state.args["_query_words"], smaller)
        self.assertEqual([item["word"] for item in remaining.state.args["items"]], smaller)
        self.assertEqual([scope["word"] for scope in remaining.state.args["_candidate_scopes"]], smaller)
        for message_id in (self.original_id, partial_id, smaller_id):
            self.assertIn(message_id, remaining.origin_message_ids)
        await self.turn("加入并提交", smaller_id)
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea"), ("小美", "xmwwi")] + [
            (word, b18.FACTS[word][0]) for word in smaller
        ], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r3_original_reply_can_select_again(self):
        await self.partial()
        await self.turn("小美 2", self.original_id)
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea"), ("小美", "xmwwi")],
                              self.replay.last_response)
        self.assertEqual(self.replay.store.get_record(self.replay.author).state.args["_query_words"],
                         [word for word in REST if word != "小美"])

    async def assert_refused(self, reply_id, message="小美 2", raw=None):
        record = self.replay.store.get_record(self.replay.author)
        snapshot, writes = copy.deepcopy(record), copy.deepcopy(self.http.writes)
        await self.turn(message, reply_id, raw=raw)
        self.assertEqual(self.http.writes, writes)
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, snapshot)
        self.assertEqual(self.replay.last_response, REFUSAL)

    async def test_r2_unrelated_bot_reply_is_refused(self):
        await REAL_FINISH(self.bot, SimpleNamespace(message_id=1), self.replay.author.actor_id,
                          self.replay.memory, "Offline unrelated reply")
        unrelated_id = self.bot.last_id
        await self.partial()
        await self.assert_refused(unrelated_id, "加入")

    async def test_r2_invalid_semantic_operands_cannot_authorize_unrelated_reply(self):
        await REAL_FINISH(self.bot, SimpleNamespace(message_id=1), self.replay.author.actor_id,
                          self.replay.memory, "Offline unrelated reply")
        unrelated_id = self.bot.last_id
        await self.partial()
        await self.assert_refused(unrelated_id, "加入", raw=b18.raw_operation(
            op="confirm", refers_to_pending=True, operands=["奈飞"], codes=["nhfw"],
        ))

    async def test_r2_older_candidate_reply_is_refused(self):
        await self.seed(("小帅", "腰线"))
        older_id = self.original_id
        await self.partial()
        await self.assert_refused(older_id, "加入")

    async def test_r2_old_partial_reply_after_ticket_replacement_is_refused(self):
        old_partial_id = await self.partial()
        await self.seed(("小美", "小帅"))
        await self.assert_refused(old_partial_id)

    async def test_r2_other_speaker_cannot_use_partial_reply(self):
        partial_id = await self.partial()
        record = self.replay.store.get_record(self.replay.author)
        snapshot, writes = copy.deepcopy(record), copy.deepcopy(self.http.writes)
        other_memory = replace(self.replay.memory, user_id="b19-other-speaker")
        self.assertTrue(self.replay.store.set(other_memory.conversation_address,
                                             b18.reviewed_ticket(("小帅",), reorder=False)))
        other_record = self.replay.store.get_record(other_memory.conversation_address)
        await REAL_FINISH(self.bot, SimpleNamespace(message_id=1), other_memory.user_id, other_memory,
                          chat._render_live_batch_record(other_record))
        other_snapshot = copy.deepcopy(other_record)
        await self.turn("加入", partial_id, "b19-other-speaker")
        self.assertEqual(self.http.writes, writes)
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, snapshot)
        self.assertIs(self.replay.store.get_record(other_memory.conversation_address), other_record)
        self.assertEqual(other_record, other_snapshot)
        self.assertEqual(self.replay.last_response, REFUSAL)


if __name__ == "__main__":
    unittest.main()
