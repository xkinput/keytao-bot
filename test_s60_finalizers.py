"""Fake-only receipt coverage, truthful no-write results, and exact batch links."""

import copy
import unittest

import test_state_machine as harness
from keytao_bot.utils.draft_receipts import (
    merge_receipt_deltas,
    receipt_change_lines,
    receipt_changes,
)


render = harness.openai_chat_module._chat_render
BATCH_ID = "s60-finalizer-batch"


def pr(identity, word, code, action="Create", **extra):
    return {"id": identity, "action": action, "word": word, "code": code,
            "type": "Phrase", **extra}


def mixed_items():
    return [pr(1, "吊杠", "dcgp"), pr(2, "掉岗", "dcgp", "Delete"),
            pr(3, "掉岗", "dcgpi"), pr(4, "沃集鲜", "wjxa")]


class ReceiptFinalizerTests(unittest.TestCase):
    def assert_exact_created_coverage(self, data, requested_words, expected_word_count):
        original = copy.deepcopy(data)
        changes = receipt_changes(data, requested_words)
        expected = {item["id"] for item in data["writtenItems"]}
        actual = [identity for change in changes for identity in change.created_ids]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected), "a created PR is omitted or counted twice")
        self.assertEqual(len(changes), expected_word_count)
        lines = receipt_change_lines(data, requested_words)
        for change in changes:
            self.assertEqual(sum(line.count(change.text) for line in lines), 1)
        self.assertEqual(data, original, "rendering mutated the actual receipt")
        return changes

    def test_one_three_four_actual_prs_create_shift_and_mixed_cover_every_id(self):
        mixed = mixed_items()
        cases = (
            (mixed[:1], ("吊杠",), 1),
            ([mixed[0], mixed[3], pr(5, "新词", "xc")], ("吊杠", "沃集鲜", "新词"), 3),
            ([mixed[0], mixed[3], pr(5, "新词", "xc"), pr(6, "另一词", "lyc")],
             ("吊杠", "沃集鲜", "新词", "另一词"), 4),
            (mixed[:3], ("吊杠",), 2),
            (mixed, ("吊杠", "沃集鲜"), 3),
        )
        for rows, requested, word_count in cases:
            with self.subTest(pr_count=len(rows), requested=requested):
                self.assert_exact_created_coverage({"success": True, "batchId": BATCH_ID,
                                                    "writtenItems": rows}, requested, word_count)

    def test_shift_pair_covers_two_prs_and_stays_next_to_requested_word(self):
        for requested, expected in (
            (("吊杠", "沃集鲜"), ["吊杠", "掉岗", "沃集鲜"]),
            (("沃集鲜", "吊杠"), ["沃集鲜", "吊杠", "掉岗"]),
        ):
            with self.subTest(requested=requested):
                changes = self.assert_exact_created_coverage(
                    {"writtenItems": mixed_items()}, requested, 3,
                )
                self.assertEqual([change.word for change in changes], expected)
                moved = next(change for change in changes if change.word == "掉岗")
                self.assertEqual(moved.text, "掉岗 dcgp→dcgpi")
                self.assertEqual(set(moved.created_ids), {2, 3})

    def test_s57_weight_changes_plus_companion_create_share_exact_coverage(self):
        rows = [pr(11, "夜泊", "yebo", "Change", oldWord="夜泊", previousWeight=101, weight=100),
                pr(12, "耶博", "yebo", "Change", oldWord="耶博", previousWeight=100, weight=101),
                pr(13, "沃集鲜", "wjxa")]
        data = {"success": True, "batchId": BATCH_ID, "writtenItems": rows,
                "shiftPlan": {"scope": "same_code", "word": "夜泊", "targetCode": "yebo"}}
        changes = self.assert_exact_created_coverage(data, ("夜泊", "耶博", "沃集鲜"), 3)
        self.assertEqual([change.text for change in changes],
                         ["夜泊 yebo 权重 101→100", "耶博 yebo 权重 100→101", "沃集鲜 → wjxa"])
        final = render.finalize_draft_receipt("✅ 操作已完成\n已变更：夜泊 yebo 权重 101→100",
                                               data, platform="qq")
        self.assertIn("耶博 yebo 权重 100→101", final)
        self.assertIn("沃集鲜 → wjxa", final)

    def test_existing_pr_updates_do_not_inflate_created_pr_count(self):
        data = {"writtenItems": [pr(4, "沃集鲜", "wjxa")],
                "updatedItems": [pr(30, "夜泊", "yebo", previousWeight=101, weight=100)]}
        changes = self.assert_exact_created_coverage(data, ("夜泊", "沃集鲜"), 2)
        self.assertEqual([identity for change in changes for identity in change.updated_ids], [30])
        self.assertEqual(changes[0].text, "夜泊 yebo 权重 101→100")

    def test_unrelated_old_snapshot_rows_and_planned_rows_cannot_become_receipt(self):
        data = {"success": True, "batchId": BATCH_ID,
                "writtenItems": [pr(4, "沃集鲜", "wjxa")],
                "draft_snapshot": {"success": True, "batchId": BATCH_ID,
                                   "items": [pr(90, "旧草稿词", "jcgc"), pr(4, "沃集鲜", "wjxa")]},
                "shiftPlan": {"word": "未写词", "targetCode": "wxc",
                              "items": [pr(91, "未写词", "wxc")]}}
        changes = self.assert_exact_created_coverage(data, ("沃集鲜",), 1)
        self.assertEqual(changes[0].word, "沃集鲜")
        final = render.finalize_draft_receipt("✅ 操作已完成\n已变更：未写词 → wxc、旧草稿词 → jcgc", data)
        self.assertIn("沃集鲜 → wjxa", final)
        self.assertNotIn("旧草稿词", final)
        self.assertNotIn("未写词", final)

    def test_missing_requested_word_retains_explicit_failure_reason(self):
        data = {"writtenItems": [pr(1, "吊杠", "dcgp")],
                "failed": [{"word": "沃集鲜", "code": "wjxa", "reason": "Candidate changed before write"}]}
        text = "\n".join(receipt_change_lines(data, ("吊杠", "沃集鲜")))
        self.assertIn("吊杠 → dcgp", text)
        self.assertIn("未新增变更：沃集鲜（Candidate changed before write）", text)
        self.assertNotIn("沃集鲜 → wjxa", text)

    def test_missing_requested_word_without_reason_stays_unknown(self):
        text = "\n".join(receipt_change_lines({"writtenItems": []}, ("沃集鲜",)))
        self.assertIn("沃集鲜", text)
        self.assertIn("原因尚未确认", text)
        self.assertNotIn("已在草稿", text)

    def test_no_write_result_removes_planned_success_claim_and_says_not_rewritten(self):
        data = {"success": True, "noWrite": True, "batchId": BATCH_ID, "writtenItems": [],
                "shiftPlan": {"word": "沃集鲜", "targetCode": "wjxa",
                              "items": [pr(40, "沃集鲜", "wjxa")]}}
        final = render.finalize_draft_receipt("✅ 操作已完成\n已变更：沃集鲜 → wjxa", data,
                                               requested_words=("沃集鲜",))
        self.assertNotIn("已变更：", final)
        self.assertNotIn("沃集鲜 → wjxa", final)
        self.assertIn("沃集鲜（已在草稿中，本轮未重复写入）", final)
        self.assertIn(f"/batch/{BATCH_ID}", final)

    def test_incomplete_delta_keeps_observed_writes_and_marks_verification_gap(self):
        data = {"writtenItems": [pr(1, "吊杠", "dcgp")], "receiptItemsUnavailable": True}
        text = "\n".join(receipt_change_lines(data, ("吊杠", "沃集鲜")))
        self.assertIn("吊杠 → dcgp", text)
        self.assertIn("本轮变更明细未能完整核验", text)
        self.assertIn("未新增变更：沃集鲜", text)

    def test_multiple_same_turn_receipts_dedupe_submit_and_keep_latest_weight(self):
        first = {"batchId": BATCH_ID, "writtenItems": mixed_items(), "requestedWords": ["吊杠", "沃集鲜"]}
        second = {"batchId": BATCH_ID, "writtenItems": copy.deepcopy(mixed_items()),
                  "updatedItems": [pr(30, "夜泊", "yebo", previousWeight=101, weight=100)]}
        third = {"batchId": BATCH_ID,
                 "updatedItems": [pr(30, "夜泊", "yebo", previousWeight=100, weight=99)]}
        original = copy.deepcopy([first, second, third])
        merged = merge_receipt_deltas([first, second, third])
        changes = self.assert_exact_created_coverage(merged, ("吊杠", "沃集鲜"), 4)
        self.assertEqual([identity for change in changes for identity in change.updated_ids], [30])
        self.assertEqual(next(change.text for change in changes if change.word == "夜泊"),
                         "夜泊 yebo 权重 101→99")
        self.assertEqual([first, second, third], original)

    def test_finalizer_combines_multiple_actual_receipts_from_same_turn(self):
        first = {"success": True, "batchId": BATCH_ID, "writtenItems": mixed_items()[:3]}
        second = {"success": True, "batchId": BATCH_ID, "writtenItems": mixed_items()[3:]}
        final = render.finalize_draft_receipt("✅ 批次已提交审核。", first, second,
                                               platform="qq", requested_words=("吊杠", "沃集鲜"))
        self.assertIn("吊杠 → dcgp", final)
        self.assertIn("掉岗 dcgp→dcgpi", final)
        self.assertIn("沃集鲜 → wjxa", final)
        self.assertNotIn("未新增变更：沃集鲜", final)

    def test_missing_url_is_reconstructed_from_exact_batch_for_each_platform(self):
        for platform, host in (("qq", "https://keytao.rea.ink"),
                               ("telegram", "https://keytao.vercel.app")):
            with self.subTest(platform=platform):
                final = render.finalize_draft_receipt("✅ 批次已提交审核。",
                                                       {"success": True, "batchId": BATCH_ID},
                                                       platform=platform)
                self.assertEqual(final.count(f"{host}/batch/{BATCH_ID}"), 1)

    def test_existing_trusted_localhost_link_survives_until_platform_delivery(self):
        local = f"http://localhost:3100/batch/{BATCH_ID}"
        source = {"success": True, "batchId": BATCH_ID, "batchUrl": local}
        for platform, host in (("qq", "https://keytao.rea.ink"),
                               ("telegram", "https://keytao.vercel.app")):
            with self.subTest(platform=platform):
                final = render.finalize_draft_receipt(f"✅ 批次已提交审核。\n草稿地址：{local}",
                                                       source, platform=platform)
                self.assertEqual(final.count(local), 1)
                delivered = render.render_platform_public_links(final, platform)
                self.assertEqual(delivered.count(f"{host}/batch/{BATCH_ID}"), 1)
                self.assertNotIn("localhost", delivered)

    def test_foreign_fallback_and_prose_batch_urls_are_replaced(self):
        foreign = "https://keytao.vercel.app/batch/foreign-batch"
        final = render.finalize_draft_receipt(f"✅ 批次已提交审核。\n草稿地址：{foreign}",
                                               {"success": True, "batchId": BATCH_ID},
                                               {"batchId": "foreign-batch", "batchUrl": foreign},
                                               platform="qq")
        self.assertNotIn(foreign, final)
        self.assertIn(f"https://keytao.rea.ink/batch/{BATCH_ID}", final)

    def test_provisional_batch_cannot_create_a_link(self):
        final = render.finalize_draft_receipt("等待确认。", {
            "requiresConfirmation": True, "batchIdProvisional": True,
            "batchId": "not-created", "batchUrl": "https://keytao.vercel.app/batch/not-created",
        }, platform="qq")
        self.assertNotIn("/batch/", final)
        self.assertIn("待确认后生成", final)


if __name__ == "__main__":
    unittest.main()
