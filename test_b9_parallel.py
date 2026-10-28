"""Offline B9 regressions for bounded, ordered per-word reviews."""

import asyncio
import json
import socket
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.utils.pending_confirmation import UNBOUND_BINDING_PRECHECK_NOTICE
from test_semantic_parser import FakeClient


class B9ParallelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.words = ("发薪", "云翻", "没头没尾", "国潮")
        self.replay = Replay({
            "sender": "b9-parallel-owner",
            "candidate_facts": {
                "发薪": {"recommended": "fsxb", "codes": ["fsxb", "fsxba"], "pinyin": "fa xin"},
                "云翻": {"recommended": "yjfj", "codes": ["yjfj", "yjfja"], "pinyin": "yun fan"},
                "没头没尾": {"recommended": "mtmw", "codes": ["mtmw", "mtmwa"], "pinyin": "mei tou mei wei"},
                "国潮": {"recommended": "gljz", "codes": ["gljz", "gljza"], "pinyin": "guo chao"},
            },
        }, harness, FakeClient)
        self.active = 0
        self.peak = 0
        self.completed = []
        self.failures = {}
        self.evidence = {}
        self.unresolved = set()
        self.replay.commands.user_resolver.reset_binding_notice_fact()
        self.addCleanup(self.replay.commands.user_resolver.reset_binding_notice_fact)
        review = self.replay.commands.keytao_review
        token = review.begin_commonness_evidence_turn()
        self.addCleanup(review.end_commonness_evidence_turn, token)
        for context in (
            patch.object(self.replay.commands, "conversation_state_store", self.replay.store),
            patch.object(self.replay.chat, "conversation_state_store", self.replay.store),
            patch.object(self.replay.commands, "call_tool_function", AsyncMock(side_effect=self.tool)),
            patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")),
        ):
            context.start()
            self.addCleanup(context.stop)

    async def tool(self, name, args, *actor, **kwargs):
        if name == "keytao_prepare_reviewed_add":
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                for _ in range(4 if args["word"] == self.words[0] else 1):
                    await asyncio.sleep(0)
                self.completed.append(args["word"])
                failure = self.failures.get(args["word"])
                if isinstance(failure, Exception):
                    raise failure
                if failure is not None:
                    return json.dumps(failure, ensure_ascii=False)
                if args["word"] in self.evidence:
                    self.replay.commands.keytao_review.record_commonness_evidence({
                        "summary": self.evidence[args["word"]],
                    })
            finally:
                self.active -= 1
        result = await self.replay.tool(name, args, *actor, **kwargs)
        if name == "keytao_prepare_reviewed_add" and args["word"] in self.unresolved:
            payload = json.loads(result)
            payload.update(pronunciationUnresolved=True, message=f"「{args['word']}」读音存在冲突，暂不推荐编码。")
            return json.dumps(payload)
        return result

    async def prepare(self):
        return await self.replay.commands._prepare_multi_word_query(
            self.words, "qq", self.replay.author.actor_id, self.replay.author,
            self.replay.author.space_key, "Offline owner", explicit_add=True,
        )

    async def test_reviews_overlap_at_most_two_and_keep_input_order(self):
        response = await self.prepare()
        self.assertEqual(self.peak, 2)
        self.assertNotEqual(self.completed[0], self.words[0])
        state = self.replay.store.get(self.replay.author)
        self.assertEqual([item["word"] for item in state.args["items"]], list(self.words))
        self.assertEqual([scope["word"] for scope in state.args["_candidate_scopes"]], list(self.words))
        positions = [response.index(f"「{word}」") for word in self.words]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.replay.writes, [])

    async def test_failed_reviews_keep_their_messages_and_other_candidates(self):
        self.failures = {
            "云翻": {"success": False, "message": "Offline review unavailable: 云翻"},
            "没头没尾": ValueError("Offline review crashed"),
        }
        response = await self.prepare()
        self.assertIn("Offline review unavailable: 云翻", response)
        self.assertIn("「没头没尾」暂时没有可核验的候选。", response)
        self.assertLess(response.index("云翻"), response.index("没头没尾"))
        state = self.replay.store.get(self.replay.author)
        self.assertEqual([item["word"] for item in state.args["items"]], ["发薪", "国潮"])
        self.assertEqual([scope["word"] for scope in state.args["_candidate_scopes"]], ["发薪", "国潮"])
        self.assertEqual(self.replay.writes, [])

    async def test_existing_and_pending_words_keep_their_own_results(self):
        self.replay.entries = [{"word": "云翻", "code": "yjfj", "type": "Phrase"}]
        self.replay.draft = [{
            "source": "draft", "batchId": "offline-batch", "batchStatus": "Draft",
            "batchUrl": "https://offline.invalid/batch/offline-batch", "itemStatus": "Pending",
            "action": "Create", "word": "没头没尾", "code": "mtmw", "type": "Phrase",
        }]
        response = await self.prepare()
        self.assertIn("已在词库：云翻 yjfj", response)
        self.assertIn("「没头没尾」已在当前草稿中", response)
        self.assertNotIn("暂时没有可核验的候选", response)
        state = self.replay.store.get(self.replay.author)
        self.assertEqual([item["word"] for item in state.args["items"]], ["发薪", "国潮"])
        self.assertEqual(self.completed, ["国潮", "发薪"])

    async def test_review_evidence_reaches_the_parent_delivery_guard_in_input_order(self):
        review = self.replay.commands.keytao_review
        review.record_commonness_evidence({"summary": "Offline existing evidence"})
        self.evidence = {
            "发薪": "Offline evidence for 发薪",
            "云翻": "Offline evidence for 云翻",
            "没头没尾": "Offline evidence for 没头没尾",
            "国潮": "Offline evidence for 国潮",
        }
        await self.prepare()
        self.assertEqual(review.current_commonness_evidence(), (
            "Offline existing evidence", "Offline evidence for 发薪", "Offline evidence for 云翻",
            "Offline evidence for 没头没尾", "Offline evidence for 国潮",
        ))

    async def test_unresolved_word_keeps_the_unbound_notice_in_the_parent(self):
        self.unresolved = {"云翻"}
        resolver = self.replay.commands.user_resolver
        with patch.object(resolver, "_find_user_payload", AsyncMock(return_value={"found": False})):
            response = await self.prepare()
        self.assertIs(resolver.resolved_binding_for_notice("qq", self.replay.author.actor_id), False)
        self.assertEqual(response.count(UNBOUND_BINDING_PRECHECK_NOTICE), 1)


if __name__ == "__main__":
    unittest.main()
