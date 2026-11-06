"""B22 community voting through real tools and offline QQ receipt turns."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import test_b19_partial_reply_binding as b19
from keytao_bot.utils import community_vote as vote
from keytao_bot.harness.orchestrator import _strip_unbound_command_suggestions
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions


chat = b19.chat
OWNER = "2416"
VOTER = "1449533601"


class VoteHTTP(b19.DraftHTTP):
    def __init__(self, batch_id):
        super().__init__()
        self.batch_id = batch_id
        self.approvals = []

    def _response(self, method, url, kwargs):
        path = urlsplit(url).path
        if path.endswith("/user/find"):
            body = json.loads(kwargs["content"]) if kwargs.get("content") else kwargs.get("json") or {}
            self.requests.append((method, path, copy.deepcopy(body)))
            return b19.b18.OfflineResponse({"found": True, "user": {"id": int(body["platformId"])}})
        if method == "GET" and path == f"/api/batches/{self.batch_id}":
            self.requests.append((method, path, {}))
            return b19.b18.OfflineResponse({"batch": {
                "id": self.batch_id, "status": "Submitted" if self.submissions else "Draft",
                "creatorId": int(OWNER if self.batch_id == "b22-owner" else VOTER),
                "contentVersion": self.version,
                "pullRequests": [{**row, "userId": int(OWNER if self.batch_id == "b22-owner" else VOTER)}
                                 for row in self.rows],
            }})
        if method == "POST" and path.endswith("/community-approve"):
            body = json.loads(kwargs["content"])
            self.approvals.append((path, body))
            return b19.b18.OfflineResponse({"success": True, "voters": ["Garth", "Two", "Three"]})
        return super()._response(method, url, kwargs)


class B22VoteTests(unittest.IsolatedAsyncioTestCase):
    tearDown = b19.B19PartialReplyBindingTests.tearDown
    matcher_send = b19.B19PartialReplyBindingTests.matcher_send
    matcher_finish = b19.B19PartialReplyBindingTests.matcher_finish
    turn = b19.B19PartialReplyBindingTests.turn

    def setUp(self):
        b19.B19PartialReplyBindingTests.setUp(self)
        self.replay.memory = replace(self.replay.memory, user_id=OWNER, space_id="865189947",
                                     speaker_name="～×！@")
        self.replay.author = self.replay.memory.conversation_address
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.votes = vote.CommunityVoteStore(Path(directory) / "votes.db")
        self.enterContext(patch.object(vote, "_default_store", self.votes))
        self.enterContext(patch.dict("os.environ", {
            "KEYTAO_COMMUNITY_VOTE": "1", "KEYTAO_PUBLIC_BASE_QQ": "https://keytao.rea.ink",
        }))
        self.enterContext(patch.dict(b19.b18.FACTS, {
            "玛卡巴卡": ("mkbkv", ["mkbkv", "mkbkva"], "ma ka ba ka"),
            "青桔": ("qkjv", ["qkjv", "qkjva"], "qing ju"),
        }))
        tools = {**b19.b18.TOOLS.TOOL_FUNCTIONS, **b19.harness._review_tools.TOOL_FUNCTIONS}
        self.enterContext(patch.object(b19.b18.commands.tool_executor, "_get_tool_function", tools.get))
        self.enterContext(patch.object(chat.skills_manager, "get_tool_function", tools.get))
        self.bot.send.side_effect = self.deliver
        self.bot.next_id = 1604886725
        self.http = VoteHTTP("b22-owner")
        self.enterContext(patch.object(b19.b18.TOOLS.http_client, "_keytao_client", self.http))

    async def deliver(self, *, event, message):
        result = await self.bot.deliver(event=event, message=message)
        await vote._record_sent_receipts(self.bot, None, "send_group_msg", {
            "group_id": self.replay.author.space_id, "message": str(message),
        }, result)
        return result

    async def submit(self, word="玛卡巴卡"):
        record = b19.b18.B18RealToolTests.seed(self, (word,), reorder=False)
        await b19.REAL_FINISH(self.bot, SimpleNamespace(message_id=1), self.replay.author.actor_id,
                             self.replay.memory, self.replay.prompt)
        candidate_id = self.bot.last_id
        await self.turn("加入", candidate_id)
        self.assertEqual([row["word"] for row in self.http.rows], [word], self.replay.last_response)
        add_id = self.bot.last_id
        await self.turn("提交", add_id, raw=b19.b18.raw_operation(op="draft_submit"))
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertIn("当前投票：0/3", self.replay.last_response)
        receipt_id = self.bot.last_id
        return receipt_id, record

    async def submitted_pair(self):
        receipt_id, _ = await self.submit()
        self.owner_http = self.http
        self.http = VoteHTTP("b22-voter")
        self.enterContext(patch.object(b19.b18.TOOLS.http_client, "_keytao_client", self.http))
        self.replay.memory = replace(self.replay.memory, user_id=VOTER, speaker_name="Garth")
        self.replay.author = self.replay.memory.conversation_address
        await self.submit("青桔")
        return receipt_id

    async def assert_vote(self, message, receipt_id, actor=VOTER, count=1):
        writes = copy.deepcopy(self.http.writes)
        raw = (b19.b18.raw_operation(op="confirm", refers_to_pending=True) if message == "同意"
               else b19.b18.raw_operation(op="chat"))
        await self.turn(message, receipt_id, actor, raw=raw)
        self.assertIn(f"当前投票：{count}/3", self.replay.last_response)
        self.assertEqual(self.http.writes, writes)
        self.assertEqual(len(self.votes.get("b22-owner")["voters"]), count)
        chat.get_ai_response_core.assert_not_awaited()

    async def test_v1_other_member_assent_on_submitted_receipt(self):
        receipt_id = await self.submitted_pair()
        own_record = self.replay.store.get_record(self.replay.author)
        snapshot = copy.deepcopy(own_record)
        await self.assert_vote("同意", receipt_id)
        self.assertEqual(self.votes.get("b22-voter")["voters"], [])
        self.assertEqual([row["platformId"] for row in self.votes.get("b22-owner")["voters"]], [VOTER])
        self.assertIs(self.replay.store.get_record(self.replay.author), own_record)
        self.assertEqual(own_record, snapshot)

    async def test_v1_receipt_send_binds_vote_round_without_stealing_candidate_ticket(self):
        receipt_id, _ = await self.submit()
        self.assertEqual(receipt_id, "1604886727")
        await self.assert_vote("同意", receipt_id)
        self.assertEqual(self.votes.resolve("qq", self.replay.author.space_id, receipt_id), "b22-owner")
        record = self.replay.store.get_record(self.replay.author)
        self.assertEqual(record.state.function_name, "keytao_submit_batch")
        self.assertNotIn(receipt_id, record.origin_message_ids)

    async def assert_alias(self, word):
        receipt_id = await self.submitted_pair()
        for index, prefix in enumerate(("", "喵喵，", "@喵喵 ", "[CQ:at,qq=3785773770] ")):
            with self.subTest(prefix=prefix):
                data = self.votes.get("b22-owner")
                if data:
                    data["voters"] = []
                    self.assertTrue(self.votes.save(data))
                await self.assert_vote(prefix + word + "！。", receipt_id, str(2000 + index))

    async def test_v2_support_votes_on_quoted_receipt(self):
        await self.assert_alias("支持")

    async def test_v2_favor_votes_on_quoted_receipt(self):
        await self.assert_alias("赞成")

    async def test_v2_vote_means_assent_without_opposing_vote_option(self):
        await self.assert_alias("投票")

    async def test_v3_self_duplicate_and_nonreceipt_votes_stay_refused(self):
        receipt_id = await self.submitted_pair()
        await self.assert_vote("支持", receipt_id)
        await self.turn("同意", receipt_id)
        self.assertEqual(self.replay.last_response, "你已经投过了（1/3）")
        await self.turn("同意", receipt_id, OWNER)
        self.assertEqual(self.replay.last_response, "提交人不能为自己投票")
        await self.turn("支持", "1604886725")
        self.assertEqual(len(self.votes.get("b22-owner")["voters"]), 1)
        self.assertEqual(self.votes.get("b22-voter")["voters"], [])
        await self.assert_vote("赞成", receipt_id, "2002", 2)
        await self.turn("投票", receipt_id, "2003")
        self.assertIn("投票通过 3/3，批次已加入词库，投票结束。", self.replay.last_response)
        self.assertEqual(self.votes.get("b22-owner")["status"], "approved")
        self.assertEqual(self.http.approvals, [(
            "/api/bot/batches/b22-owner/community-approve", {
                "platform": "qq", "platformId": OWNER, "expectedContentVersion": self.owner_http.version,
                "voters": [{"platform": "qq", "platformId": actor} for actor in (VOTER, "2002", "2003")],
            },
        )])


class B22StripperTests(unittest.TestCase):
    def test_v4_facts_before_addressed_command_are_preserved(self):
        for facts, command, suggestions in (
            ("已记录玛卡巴卡 mkbkv。", "@Garth 你可以回复「提交」提交。", ("提交",)),
            ("当前投票 1/3。", "@Rea 请引用回执回复「同意」投票。", ("同意",)),
        ):
            with self.subTest(facts=facts):
                self.assertEqual(
                    _strip_unbound_command_suggestions(facts + command, suggestions), facts,
                )

    def test_v4_unclosed_parenthesis_preserves_facts_before_command(self):
        for facts, command in (
            ("已加入草稿：玛卡巴卡 mkbkv（读音 mǎ kǎ bā kǎ。草稿已保存。", "回复「提交」提交。"),
            ("玛卡巴卡（mkbkv 已在草稿中。", "可以回复「提交」提交批次。"),
            ("已加入草稿：玛卡巴卡 mkbkv(读音 mǎ kǎ bā kǎ。草稿已保存。", "回复「提交」提交。"),
            ("玛卡巴卡(mkbkv 已在草稿中。", "可以回复「提交」提交批次。"),
        ):
            with self.subTest(facts=facts):
                self.assertEqual(
                    _strip_unbound_command_suggestions(facts + command, ("提交",)), facts,
                )

    def test_v4_model_vote_prose_never_leaves_half_deleted_sentence(self):
        facts = ("没有执行写入：本轮未取得可信的投票结果。\n"
                 "- 被引用批次：「玛卡巴卡（→ mkbkv）」，当前已知投票为 0/3。")
        for nickname in ("（～×！@）", "(～×!@)", "（～×？@）", "(～×?@)",
                         "～×！@", "～×!@", "～×？@", "～×?@"):
            for separator in ("\n", ""):
                with self.subTest(nickname=nickname, separator=separator):
                    command = ("请回复「同意」支持该批次；该批次由" + nickname
                               + "本人提交，若规则不允许提交者参与投票，请由其他成员引用投票。")
                    text = facts + separator + command
                    suggestions = advertised_command_suggestions(text)
                    self.assertIn("同意", suggestions)
                    self.assertEqual(_strip_unbound_command_suggestions(text, suggestions), facts)


if __name__ == "__main__":
    unittest.main()
