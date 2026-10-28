"""Offline B9 regressions for adding words while a submit ticket is live."""

import copy
import json
import socket
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.harness.state import PendingAddWord, PendingToolConfirm
from keytao_bot.utils import http_client
from test_semantic_parser import FakeClient, raw_operation


class AddKeepsSubmitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.replay = Replay({
            "sender": "b9-owner",
            "draft": [{"id": 1, "action": "Create", "word": "驴友", "code": "llyd", "type": "Phrase"}],
            "candidate_facts": {
                "屡有": {"recommended": "llydav", "codes": ["llydav"], "pinyin": "lv you"},
                "云翻": {"recommended": "yjfj", "codes": ["yjfj"], "pinyin": "yun fan"},
            },
        }, harness, FakeClient)
        self.old_draft = copy.deepcopy(self.replay.draft)
        self.foreign = False
        self.submitted = []
        self.original_tool = self.replay.tool
        self.replay.tool = self.tool
        self.details = AsyncMock(side_effect=lambda *args, **kwargs: {"batch": {
            "id": "offline-batch", "contentVersion": 1, "status": "Draft", "creatorId": 17,
            "pullRequests": [{**item, "userId": 99 if self.foreign and item["id"] == 1 else 17}
                             for item in self.replay.draft],
        }})
        stack = ExitStack()
        self.addCleanup(stack.close)
        for context in self.replay.patches():
            stack.enter_context(context)
        stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        stack.enter_context(patch.object(http_client, "keytao_json", self.details))
        stack.enter_context(patch.dict("os.environ", {"KEYTAO_COMMUNITY_VOTE": "0"}))
        token = self.replay.chat.current_draft_delivery_claims.set([])
        self.addCleanup(self.replay.chat.current_draft_delivery_claims.reset, token)

    def preview(self):
        return self.replay.result(
            success=False, requiresConfirmation=True, snapshotItems=copy.deepcopy(self.replay.draft),
            snapshotDigest="a" * 64, warningDigest="b" * 64, auditDigest="c" * 64, warnings=[],
        )

    async def tool(self, name, args, platform, user_id, **kwargs):
        if name != "keytao_submit_batch":
            return await self.original_tool(name, args, platform, user_id, **kwargs)
        self.assertEqual((platform, user_id), ("qq", self.replay.author.actor_id))
        self.replay.calls.append((name, copy.deepcopy(args)))
        for index, item in enumerate(self.replay.draft, 1):
            item.setdefault("id", index)
        if not args.get("confirmed"):
            return json.dumps(self.preview())
        self.assertEqual(args, {
            "batch_id": "offline-batch", "confirmed": True, "expected_content_version": 1,
            "expected_server_snapshot_digest": "a" * 64,
            "expected_warning_digest": "b" * 64, "expected_audit_digest": "c" * 64,
        })
        self.replay.writes.append((name, copy.deepcopy(args)))
        self.submitted.append(copy.deepcopy(self.replay.draft))
        return json.dumps(self.replay.result(snapshotItems=copy.deepcopy(self.replay.draft)))

    async def offer(self, words=("屡有", "云翻")):
        replay = self.replay
        pending = replay.commands._pending_state_from_server_warning(
            PendingToolConfirm("keytao_submit_batch", {"batch_id": "offline-batch"}), self.preview(),
        )
        self.assertTrue(replay.store.set(replay.author, pending))
        replay.history[:] = [{"role": "assistant", "content": "回复「确认」提交，或「取消」。"}]
        await replay.turn({"message": "加词：" + "，".join(words), "to_me": True}, raw_operation(
            op="add", operands=list(words), codes=[], frame="direct", refers_to_pending=False,
        ))
        self.assertIn("加入并提交", replay.last_response)
        self.assertEqual(replay.writes, [])
        self.assertEqual(replay.draft, self.old_draft)
        return replay.store.get_record(replay.author)

    async def answer(self, message):
        replay = self.replay
        return await replay.commands.handle_pending_message_core(
            message, "qq", replay.author.actor_id, replay.author,
            space_key=replay.author.space_key, allow_intent_model=False,
        )

    async def test_single_and_multi_candidates_keep_submit_through_delivery(self):
        for words in (("屡有",), ("屡有", "云翻")):
            with self.subTest(words=words):
                record = await self.offer(words)
                self.assertTrue(isinstance(record.state, PendingAddWord) or
                                record.state.function_name == "keytao_batch_add_to_draft")
                self.assertNotIn("原待确认提交已取消", self.replay.last_response)
                delivered = self.replay.chat._prepare_user_facing_reply(self.replay.last_response, self.replay.memory)
                self.assertNotIn("原待确认提交已取消", delivered)
                self.assertTrue(self.replay.chat._advertised_reply_matches_live_record(delivered, record))
                self.assertEqual(self.replay.chat.current_draft_delivery_claims.get(), [])

    async def test_add_and_submit_includes_own_old_items_once(self):
        for words in (("屡有",), ("屡有", "云翻")):
            with self.subTest(words=words):
                self.replay.draft = copy.deepcopy(self.old_draft)
                self.replay.writes.clear()
                await self.offer(words)
                response = await self.answer("加入并提交")
                self.assertIn("已提交审核", response)
                self.assertEqual([item["word"] for item in self.submitted[-1]], ["驴友", *words])
                self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in self.replay.writes), 1)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_foreign_old_items_still_need_scope_confirmation(self):
        self.foreign = True
        await self.offer()
        response = await self.answer("加入并提交")
        self.assertIn("本轮之外的内容", response)
        self.assertIn("确认", response)
        self.assertEqual(self.submitted, [])
        state = self.replay.store.get(self.replay.author)
        self.assertEqual(state.function_name, "keytao_submit_batch")
        self.assertIn("已提交审核", await self.answer("确认"))
        self.assertEqual([item["word"] for item in self.submitted[0]], ["驴友", "屡有", "云翻"])

    async def test_cancel_new_candidates_leaves_old_draft_untouched(self):
        await self.offer()
        self.replay.calls.clear()
        self.assertEqual(await self.answer("取消"), "已取消。")
        self.assertEqual(self.replay.draft, self.old_draft)
        self.assertEqual(self.replay.calls, [])
        self.assertEqual(self.replay.writes, [])
        self.assertIsNone(self.replay.store.get_record(self.replay.author))


if __name__ == "__main__":
    unittest.main()
