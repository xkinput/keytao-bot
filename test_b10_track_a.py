"""Offline regressions for plus-one filtering and QQ receipt links."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_state_machine as harness
from keytao_bot.utils import community_vote as vote
from test_community_vote import submitted_data


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


if __name__ == "__main__":
    unittest.main()
