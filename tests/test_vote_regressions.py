"""Topic regressions: vote regressions."""

import unittest as _stage_t_unittest
from tests.helpers import (
    B19PartialReplyBindingTests_matcher_finish,
    B19PartialReplyBindingTests_matcher_send,
    B19PartialReplyBindingTests_tearDown,
    B19PartialReplyBindingTests_turn,
    B22VoteTests_assert_alias,
    B22VoteTests_assert_vote,
    B22VoteTests_deliver,
    B22VoteTests_setUp,
    B22VoteTests_submit,
    B22VoteTests_submitted_pair,
    OWNER,
    VOTER,
    _strip_unbound_command_suggestions,
    advertised_command_suggestions,
    copy,
    harness,
    os,
    patch,
    snapshot,
    submitted_data,
    subprocess,
    sys,
    unittest,
    vote,
)


# Source: test_b10_track_a.py
class PlusOneGuardTests(unittest.TestCase):
    def test_only_bot_directed_messages_and_commands_are_ignored(self):
        # The existing harness replaces NoneBot; check registration in a clean process.
        result = subprocess.run([sys.executable, "-c", """
import asyncio
from types import SimpleNamespace as NS
from nonebot.exception import IgnoredException
from nonebot.message import _run_preprocessors
from nonebot.adapters.onebot.v11 import Message, MessageSegment
from keytao_bot.plugins.plus_one_guard import guard_plus_one

assert any(proc.call is guard_plus_one for proc in _run_preprocessors)
bot = NS(self_id="42")
plus = NS(plugin_name="nonebot_plugin_plus_one")
other = NS(plugin_name="openai_chat")

def event(*, to_me=False, at=None, reply=None):
    return NS(
        is_tome=lambda: to_me, get_message=lambda: Message("same request"),
        original_message=(MessageSegment.at(at) + "same request") if at else Message("same request"),
        reply=NS(sender=NS(user_id=reply)) if reply else None,
    )

async def check():
    for incoming, state in (
        (event(to_me=True), {}),
        (event(at="42"), {}),
        (event(reply=42), {}),
        (event(), {"_prefix": {"command": ("clear",)}}),
    ):
        try:
            await guard_plus_one(bot, incoming, plus, state)
        except IgnoredException:
            pass
        else:
            raise AssertionError("Bot message reached plus-one")
        await guard_plus_one(bot, incoming, other, state)
    for incoming in (event(), event(at="99"), event(reply=99)):
        await guard_plus_one(bot, incoming, plus, {"_prefix": {"command": None}})

asyncio.run(check())
"""], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

# Source: test_b10_track_a.py
class QQReceiptLinkTests(unittest.TestCase):
    def test_vote_receipt_uses_platform_base_at_final_delivery(self):
        memory = harness.ChatMemoryContext(
            platform="qq", user_id="b10-owner", space_type="group", space_id="b10-group",
        )
        data = submitted_data()
        data["batchUrl"] = data["batchUrl"].replace("keytao.rea.ink", "keytao.vercel.app")
        token = vote._pending_receipts.set([])
        self.addCleanup(vote._pending_receipts.reset, token)
        with patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": "1"}), patch.object(vote, "install_delivery_hook"):
            receipt = vote.submitted_vote_receipt(data)
            for base in ("https://keytao.rea.ink", "https://qq.example.test"):
                with self.subTest(base=base), patch.dict(os.environ, {"KEYTAO_PUBLIC_BASE_QQ": base}):
                    delivered = harness.openai_chat_module._prepare_user_facing_reply(receipt, memory)
                    self.assertIn(f"{base}/batch/{data['batchId']}", delivered)
                    self.assertNotIn("keytao.vercel.app", delivered)
                    self.assertIn("0/3", delivered)
                    self.assertEqual(harness.openai_chat_module._prepare_user_facing_reply(delivered, memory), delivered)

# Source: test_b22_vote.py
class B22VoteTests(unittest.IsolatedAsyncioTestCase):
    tearDown = B19PartialReplyBindingTests_tearDown
    matcher_send = B19PartialReplyBindingTests_matcher_send
    matcher_finish = B19PartialReplyBindingTests_matcher_finish
    turn = B19PartialReplyBindingTests_turn

    setUp = B22VoteTests_setUp

    deliver = B22VoteTests_deliver

    submit = B22VoteTests_submit

    submitted_pair = B22VoteTests_submitted_pair

    assert_vote = B22VoteTests_assert_vote

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

    assert_alias = B22VoteTests_assert_alias

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

# Source: test_b22_vote.py
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


def load_tests(loader, standard_tests, pattern):
    return _stage_t_unittest.TestSuite(
        loader.loadTestsFromTestCase(case) for case in (
            PlusOneGuardTests,
            QQReceiptLinkTests,
            B22VoteTests,
            B22StripperTests,
        )
    )


if __name__ == "__main__":
    _stage_t_unittest.main()
