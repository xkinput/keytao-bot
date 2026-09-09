"""Offline S60 regressions: receipts cover the actual turn delta."""

import asyncio
import copy
import json
import unittest
from unittest.mock import patch

import test_state_machine as harness

commands = harness.chat_commands_module
chat = harness.openai_chat_module


def mixed_receipt():
    items = [
        {"id": "pr-1", "action": "Create", "word": "吊杠", "code": "dcgp", "type": "Phrase"},
        {"id": "pr-2", "action": "Delete", "word": "掉岗", "code": "dcgp", "type": "Phrase"},
        {"id": "pr-3", "action": "Create", "word": "掉岗", "code": "dcgpi", "type": "Phrase"},
        {"id": "pr-4", "action": "Create", "word": "沃集鲜", "code": "wjxa", "type": "Phrase"},
    ]
    return {
        "success": True, "batchId": "e4d21c4d", "pullRequestCount": 4,
        "writtenItems": items,
        "shiftPlan": {
            "word": "吊杠", "targetCode": "dcgp", "items": copy.deepcopy(items),
            "shifted": [{"word": "掉岗", "fromCode": "dcgp", "toCode": "dcgpi"}],
        },
    }


class ReceiptRegressionTests(unittest.TestCase):
    def test_orchestrator_normal_and_failure_tails_keep_the_actual_delta(self):
        from keytao_bot.harness.orchestrator import AgentOrchestrator
        actual = mixed_receipt()
        actual["requestedWords"] = ["吊杠", "沃集鲜"]
        receipt = AgentOrchestrator._successful_write_receipt("keytao_shift_phrase_code", {}, actual)
        submitted = AgentOrchestrator._successful_write_receipt("keytao_submit_batch", {}, {
            "success": True, "batchId": actual["batchId"], "writtenItems": [], "updatedItems": [],
        })
        for failure, termination, receipts in (
            ({}, {}, [receipt, submitted]),
            ({"message": "Fixture submit failed", "_failedTool": "keytao_submit_batch"}, {}, [receipt]),
            ({}, {"reason": "transport_failure"}, [receipt]),
            ({}, {"reason": "iteration_cap"}, [receipt]),
        ):
            with self.subTest(failure=failure, termination=termination):
                reply = AgentOrchestrator._finalize_reply("加入并提交", "已变更：吊杠 → dcgp", failure, receipts, termination)
                for expected in ("吊杠 → dcgp", "掉岗 dcgp→dcgpi", "沃集鲜 → wjxa", "/batch/e4d21c4d"):
                    self.assertIn(expected, reply)
                self.assertEqual(reply.count("已提交审核"), 1 if submitted in receipts else 0)

    def test_final_delivery_recovers_rows_and_link_after_a_failed_summary(self):
        from keytao_bot.utils.memory_store import ChatMemoryContext
        context = ChatMemoryContext(platform="qq", user_id="s60-delivery", space_type="private", space_id="s60-delivery")
        token = commands.current_draft_delivery_claims.set([])
        try:
            data = mixed_receipt()
            data["requestedWords"] = ["沃集鲜", "吊杠"]
            commands._capture_successful_draft_write_delivery("keytao_shift_phrase_code", data, "qq", "s60-delivery")
            with patch.object(chat, "conversation_state_store", harness.MemoryConversationStateStore()):
                reply = chat._prepare_user_facing_reply("这次处理没有完成；本次未执行新的写入。", context)
            self.assertIn("沃集鲜 → wjxa", reply)
            self.assertIn("掉岗 dcgp→dcgpi", reply)
            self.assertNotIn("本次未执行新的写入", reply)
            self.assertLess(reply.index("沃集鲜"), reply.index("吊杠"))
            self.assertEqual(reply.count("https://keytao.rea.ink/batch/e4d21c4d"), 1)
            for approved in (False, True):
                commands.current_draft_delivery_claims.get()[:] = []
                commands._capture_successful_draft_write_delivery("keytao_shift_phrase_code", data, "qq", "s60-delivery")
                commands._capture_successful_draft_write_delivery("keytao_submit_batch", {
                    "success": True, "batchId": data["batchId"], "writtenItems": [], "updatedItems": [],
                    "autoApproved": approved,
                }, "qq", "s60-delivery")
                with patch.object(chat, "conversation_state_store", harness.MemoryConversationStateStore()):
                    submitted = chat._prepare_user_facing_reply("本次未写入。", context)
                self.assertIn("已加入词库" if approved else "已提交审核", submitted)
                self.assertIn("沃集鲜 → wjxa", submitted)
                self.assertNotIn("后续处理未完成", submitted)
            other = ChatMemoryContext(platform="qq", user_id="another-actor", space_type="private", space_id="another-actor")
            with patch.object(chat, "conversation_state_store", harness.MemoryConversationStateStore()):
                self.assertNotIn("e4d21c4d", chat._prepare_user_facing_reply("查询完成", other))
        finally:
            commands.current_draft_delivery_claims.reset(token)

    def test_plan_order_cannot_override_user_order_at_shared_finalizer(self):
        from keytao_bot.plugins.chat_render import finalize_draft_receipt
        data = mixed_receipt()
        data["requestedItems"] = data["shiftPlan"]["items"]
        data["requestedWords"] = ["沃集鲜", "吊杠"]
        reply = finalize_draft_receipt("✅ 操作已完成", data, platform="qq")
        self.assertLess(reply.index("沃集鲜"), reply.index("吊杠"))

    def test_no_write_reason_stays_with_its_own_word(self):
        from keytao_bot.utils.draft_receipts import merge_receipt_deltas, receipt_change_lines
        merged = merge_receipt_deltas([
            {"writtenItems": [], "noWrite": True, "requestedWords": ["吊杠"]},
            {"writtenItems": [], "receiptItemsUnavailable": True, "requestedWords": ["沃集鲜"]},
        ])
        lines = receipt_change_lines(merged)
        self.assertIn("未新增变更：吊杠（已在草稿中，本轮未重复写入）", lines)
        unknown = next(line for line in lines if line.startswith("未新增变更：沃集鲜"))
        self.assertIn("原因尚未确认", unknown)
        self.assertNotIn("已在草稿", unknown)

    def test_mixed_shift_success_keeps_the_companion_create(self):
        reply = commands._format_ranked_shift_success(mixed_receipt())
        self.assertIn("吊杠 → dcgp", reply)
        self.assertIn("掉岗 dcgp→dcgpi", reply)
        self.assertIn("沃集鲜 → wjxa", reply)

    def test_submit_without_url_keeps_exact_platform_batch_link(self):
        async def run():
            for platform, base in (("qq", "https://keytao.rea.ink"), ("telegram", "https://keytao.vercel.app")):
                with self.subTest(platform=platform), patch.object(
                    commands, "call_tool_function", return_value=json.dumps({
                        "success": True, "batchId": "e4d21c4d", "status": "Submitted", "contentVersion": 2,
                    }),
                ):
                    result = await commands._perform_submit_current_draft(platform, "s60-offline", confirmed=True)
                    self.assertIn(base + "/batch/e4d21c4d", result.text)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
