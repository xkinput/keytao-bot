"""S60 actual-receipt deltas use fake snapshots and never dispatch provider I/O."""

import copy
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from keytao_bot.utils import completed_draft_undo as journal


def row(item_id, word, code, action="Create", weight=10):
    return {"id": item_id, "word": word, "code": code, "type": "Phrase",
            "action": action, "oldWord": None, "weight": weight,
            "remark": None, "needsManualReview": False}


OLD = row(11, "旧词", "jqck")
CREATES = [row(12, "吊杠", "dcgp"), row(13, "掉岗", "dcgp", "Delete"),
           row(14, "掉岗", "dcgpi"), row(15, "沃集鲜", "wjxa")]


def snapshot(items, version=1):
    return {"batchId": "s60-delta", "contentVersion": version,
            "batchUrl": "https://example.invalid/batch/s60-delta", "items": copy.deepcopy(items)}


class ReceiptDeltaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.capture = journal.OperationCapture("s60-scope", "qq", "fixture-actor", "s60-turn", "")
        self.token = journal.current_operation_capture.set(self.capture)
        self.store = MagicMock()
        self.store.actor_has_running_undo.return_value = False
        self.store_patch = patch.object(journal, "get_default_draft_mutation_claim_store", return_value=self.store)
        self.store_patch.start()
        self.listed = AsyncMock(side_effect=AssertionError("unexpected real snapshot read"))
        self.get_tool = MagicMock(return_value=self.listed)

    def tearDown(self):
        self.store_patch.stop()
        journal.current_operation_capture.reset(self.token)

    async def invoke(self, before, after, *, name="keytao_batch_add_to_draft", args=None, result=None):
        args = args if args is not None else {"confirmed": True, "items": copy.deepcopy(CREATES)}
        result = result if result is not None else {
            "success": True, "batchId": "s60-delta",
            "contentVersion": after["contentVersion"] if isinstance(after, dict) else 2,
        }
        tool = AsyncMock(return_value=result)
        with patch.object(journal, "_read_snapshot", AsyncMock(side_effect=[before, after])) as read:
            returned = await journal.invoke_with_operation_journal(name, args, tool, self.get_tool)
        return returned, read, tool

    async def test_one_three_and_four_actual_pr_rows_exclude_old_unrelated_ids(self):
        for items in ([CREATES[3]], CREATES[:3], CREATES):
            with self.subTest(count=len(items)):
                result, read, tool = await self.invoke(snapshot([OLD]), snapshot([OLD, *items], 2))
                self.assertEqual(result["writtenItems"], items)
                self.assertEqual(len(result["writtenItems"]), len(items))
                self.assertEqual(result["updatedItems"], [])
                self.assertEqual(read.await_count, 2)
                tool.assert_awaited_once()
                self.assertNotIn("receiptItemsUnavailable", result)

    async def test_shift_requested_items_cannot_become_missing_actual_writes(self):
        original = {"success": True, "batchId": "s60-delta", "pullRequestCount": 3, "contentVersion": 2,
                    "shiftPlan": {"items": copy.deepcopy(CREATES)}}
        result, _, _ = await self.invoke(
            snapshot([OLD]), snapshot([OLD, *CREATES[:3]], 2),
            name="keytao_shift_phrase_code", args={"confirmed_plan_digest": "a" * 64}, result=original,
        )
        self.assertEqual(result["writtenItems"], CREATES[:3])
        self.assertEqual(result["requestedItems"], CREATES)
        self.assertNotIn("writtenItems", original)

    async def test_partial_write_preserves_failure_and_only_reports_verified_rows(self):
        result, read, _ = await self.invoke(
            snapshot([OLD]), snapshot([OLD, CREATES[0]], 2),
            result={"success": False, "partialWrite": True, "batchId": "s60-delta", "contentVersion": 2,
                    "pullRequestCount": 1, "message": "fixture partial"},
        )
        self.assertIs(result["success"], False)
        self.assertEqual(result["writtenItems"], CREATES[:1])
        self.assertEqual(result["requestedItems"], CREATES)
        self.assertEqual(read.await_count, 2)
        self.store.save_completed_operation.assert_not_called()

    async def test_multiple_invocations_report_disjoint_deltas_with_cumulative_undo(self):
        first_before = snapshot([OLD])
        first_after = snapshot([OLD, CREATES[0]], 2)
        second_after = snapshot([OLD, CREATES[0], CREATES[3]], 3)
        first, _, _ = await self.invoke(first_before, first_after)
        second, _, _ = await self.invoke(first_after, second_after)
        self.assertEqual(first["writtenItems"], [CREATES[0]])
        self.assertEqual(second["writtenItems"], [CREATES[3]])
        self.assertEqual(self.capture.before, first_before)
        self.assertEqual(self.capture.payload["after"], second_after)
        self.assertEqual(self.store.save_completed_operation.call_count, 2)

    async def test_submit_adds_no_pr_rows_and_performs_no_post_submit_read(self):
        after_create = snapshot([OLD, *CREATES], 2)
        await self.invoke(snapshot([OLD]), after_create)
        result, read, _ = await self.invoke(
            after_create, AssertionError("post-submit read"), name="keytao_submit_batch",
            args={"confirmed": True, "batch_id": "s60-delta"},
            result={"success": True, "batchId": "s60-delta", "status": "Submitted", "contentVersion": 3},
        )
        self.assertEqual(result["writtenItems"], [])
        self.assertEqual(result["updatedItems"], [])
        self.assertEqual(read.await_count, 1)
        self.assertTrue(self.capture.payload["submitted"])
        self.assertEqual(self.capture.payload["after"]["items"], after_create["items"])
        self.assertEqual(self.capture.payload["after"]["contentVersion"], 3)

    async def test_weight_update_keeps_actual_nullable_previous_weight_separate_from_creates(self):
        for previous in (None, 0, 10):
            with self.subTest(previous=previous):
                original = {**OLD, "weight": previous}
                changed = {**OLD, "weight": 20}
                result, _, _ = await self.invoke(
                    snapshot([original]), snapshot([changed], 2),
                    name="keytao_update_draft_item_weight",
                    args={"word": OLD["word"], "code": OLD["code"], "weight": 20},
                )
                self.assertEqual(result["writtenItems"], [])
                self.assertEqual(result["updatedItems"], [{**changed, "previousWeight": previous}])
                result, _, _ = await self.invoke(
                    snapshot([original]), snapshot([changed, CREATES[3]], 2),
                    name="keytao_shift_phrase_code", args={"confirmed_plan_digest": "a" * 64},
                    result={"success": True, "batchId": "s60-delta", "contentVersion": 2, "pullRequestCount": 1,
                            "shiftPlan": {"items": [CREATES[3]], "draftUpdates": [{
                                "word": OLD["word"], "code": OLD["code"], "fromWeight": previous, "toWeight": 20,
                            }]}},
                )
                self.assertEqual(result["writtenItems"], [CREATES[3]])
                self.assertEqual(result["updatedItems"], [{**changed, "previousWeight": previous}])

    async def test_cross_batch_or_missing_actual_batch_id_never_claims_delta(self):
        cases = (
            (snapshot([OLD]), {**snapshot([OLD, CREATES[0]], 2), "batchId": "another-batch"}, "s60-delta"),
            ({**snapshot([OLD]), "batchId": "another-batch"}, snapshot([OLD, CREATES[0]], 2), "s60-delta"),
            (snapshot([OLD]), snapshot([OLD, CREATES[0]], 2), None),
            (snapshot([OLD]), snapshot([OLD, CREATES[0]], 2), ""),
        )
        for before, after, batch_id in cases:
            with self.subTest(batch_id=batch_id, before=before["batchId"], after=after["batchId"]):
                result, _, _ = await self.invoke(before, after, result={
                    "success": True, "batchId": batch_id, "contentVersion": 2, "pullRequestCount": 1,
                })
                self.assertEqual(result["writtenItems"], [])
                self.assertEqual(result["updatedItems"], [])
                self.assertTrue(result["receiptItemsUnavailable"])

    async def test_verified_empty_absence_may_become_new_batch(self):
        before = {"batchId": "", "contentVersion": 0, "items": []}
        result, _, _ = await self.invoke(before, snapshot([CREATES[0]], 1), result={
            "success": True, "batchId": "s60-delta", "contentVersion": 1, "pullRequestCount": 1,
        })
        self.assertEqual(result["writtenItems"], [CREATES[0]])
        self.assertNotIn("receiptItemsUnavailable", result)

    async def test_omitted_request_type_does_not_invent_phrase_type(self):
        actual = {**CREATES[0], "type": "Single"}
        result, _, _ = await self.invoke(
            snapshot([OLD]), snapshot([OLD, actual], 2), name="keytao_create_phrase",
            args={"confirmed": True, "word": actual["word"], "code": actual["code"]},
            result={"success": True, "batchId": "s60-delta", "contentVersion": 2, "pullRequestCount": 1},
        )
        self.assertEqual(result["writtenItems"], [actual])
        self.assertNotIn("receiptItemsUnavailable", result)

    async def test_later_snapshot_and_wrong_created_count_are_unavailable(self):
        for version, count in ((2, 1), (3, 4)):
            with self.subTest(version=version, count=count):
                result, _, _ = await self.invoke(snapshot([OLD]), snapshot([OLD, *CREATES[:3]], 3), result={
                    "success": True, "batchId": "s60-delta", "contentVersion": version, "pullRequestCount": count,
                })
                self.assertEqual(result["writtenItems"], [])
                self.assertEqual(result["updatedItems"], [])
                self.assertTrue(result["receiptItemsUnavailable"])

    async def test_exact_count_cannot_authenticate_extra_update_even_with_matching_version(self):
        for receipt_version in ({}, {"contentVersion": 2}):
            with self.subTest(receipt_version=receipt_version):
                result, _, _ = await self.invoke(
                    snapshot([OLD]), snapshot([{**OLD, "weight": 99}, CREATES[0]], 2),
                    result={"success": True, "batchId": "s60-delta", "pullRequestCount": 1, **receipt_version},
                )
                self.assertEqual(result["writtenItems"], [])
                self.assertEqual(result["updatedItems"], [])
                self.assertTrue(result["receiptItemsUnavailable"])

    async def test_missing_version_needs_exact_count_requested_rows_and_unchanged_old_rows(self):
        good, _, _ = await self.invoke(snapshot([OLD]), snapshot([OLD, CREATES[0]], 2), result={
            "success": True, "batchId": "s60-delta", "pullRequestCount": 1,
        })
        self.assertEqual(good["writtenItems"], [CREATES[0]])
        for changes in ({"word": "别人的词"}, {"type": "Single"}, {"oldWord": "别名"}, {"weight": 99}):
            with self.subTest(changes=changes):
                result, _, _ = await self.invoke(
                    snapshot([OLD]), snapshot([OLD, {**CREATES[0], **changes}], 2),
                    result={"success": True, "batchId": "s60-delta", "pullRequestCount": 1},
                )
                self.assertEqual(result["writtenItems"], [])
                self.assertTrue(result["receiptItemsUnavailable"])
        removed, _, _ = await self.invoke(snapshot([OLD]), snapshot([CREATES[0]], 2), result={
            "success": True, "batchId": "s60-delta", "pullRequestCount": 1,
        })
        self.assertTrue(removed["receiptItemsUnavailable"])
        unknown, _, _ = await self.invoke(snapshot([OLD]), snapshot([OLD, CREATES[0]], 2), result={
            "success": True, "batchId": "s60-delta",
        })
        self.assertTrue(unknown["receiptItemsUnavailable"])

    async def test_no_write_retains_full_requested_labels(self):
        result, _, _ = await self.invoke(
            snapshot([OLD, *CREATES]), None, name="keytao_shift_phrase_code",
            args={"confirmed_plan_digest": "a" * 64},
            result={"success": True, "noWrite": True, "shiftPlan": {"items": [], "plannedItems": CREATES}},
        )
        self.assertEqual(result["requestedItems"], CREATES)
        self.assertEqual(result["writtenItems"], [])

    async def test_server_review_metadata_does_not_hide_acknowledged_prs(self):
        for versioned in (False, True):
            actual = {**CREATES[0], "remark": "Server audit evidence", "needsManualReview": True}
            result, _, _ = await self.invoke(snapshot([OLD]), snapshot([OLD, actual], 2), result={
                "success": True, "batchId": "s60-delta", "pullRequestCount": 1,
                **({"contentVersion": 2} if versioned else {}),
            })
            self.assertEqual(result["writtenItems"], [actual])
            self.assertNotIn("receiptItemsUnavailable", result)

    async def test_no_write_and_replays_erase_stale_delta_without_post_read_or_journal_save(self):
        for flag in ("noWrite", "alreadyApplied", "replayedResolvedMutation"):
            with self.subTest(flag=flag):
                result, read, _ = await self.invoke(
                    snapshot([OLD, *CREATES]), None,
                    result={"success": True, flag: True, "writtenItems": CREATES,
                            "updatedItems": [OLD], "receiptItemsUnavailable": True},
                )
                self.assertEqual(result["writtenItems"], [])
                self.assertEqual(result["updatedItems"], [])
                self.assertNotIn("receiptItemsUnavailable", result)
                self.assertEqual(read.await_count, 1)
        self.store.save_completed_operation.assert_not_called()

    async def test_missing_post_snapshot_never_promotes_count_or_plan_to_written_rows(self):
        result, read, _ = await self.invoke(
            snapshot([OLD]), None,
            result={"success": True, "batchId": "s60-delta", "pullRequestCount": 4, "writtenItems": CREATES},
        )
        self.assertEqual(result["writtenItems"], [])
        self.assertEqual(result["updatedItems"], [])
        self.assertTrue(result["receiptItemsUnavailable"])
        self.assertEqual(read.await_count, 2)
        self.assertIsNone(self.capture.payload["after"])

    async def test_missing_capture_marks_success_unverified_without_extra_reads(self):
        journal.current_operation_capture.set(None)
        tool = AsyncMock(return_value={"success": True, "batchId": "s60-delta", "pullRequestCount": 4})
        with patch.object(journal, "_read_snapshot", AsyncMock()) as read:
            result = await journal.invoke_with_operation_journal(
                "keytao_batch_add_to_draft", {"confirmed": True, "items": CREATES}, tool, self.get_tool,
            )
        self.assertEqual(result["writtenItems"], [])
        self.assertTrue(result["receiptItemsUnavailable"])
        read.assert_not_awaited()

    async def test_missing_capture_preserves_normalized_tool_receipt_but_never_replayed_rows(self):
        journal.current_operation_capture.set(None)
        original = {"success": True, "batchId": "s60-delta", "writtenItems": CREATES,
                    "updatedItems": [{**OLD, "weight": 20, "previousWeight": 10}]}
        with patch.object(journal, "_read_snapshot", AsyncMock()) as read:
            result = await journal.invoke_with_operation_journal(
                "keytao_batch_add_to_draft", {"confirmed": True, "items": CREATES},
                AsyncMock(return_value=original), self.get_tool,
            )
            self.assertEqual(result["writtenItems"], original["writtenItems"])
            self.assertEqual(result["updatedItems"], original["updatedItems"])
            self.assertNotIn("receiptItemsUnavailable", result)
            for flag in ("noWrite", "alreadyApplied", "replayedResolvedMutation"):
                cleared = await journal.invoke_with_operation_journal(
                    "keytao_batch_add_to_draft", {"confirmed": True, "items": CREATES},
                    AsyncMock(return_value={**original, flag: True}), self.get_tool,
                )
                self.assertEqual(cleared["writtenItems"], [])
                self.assertEqual(cleared["updatedItems"], [])
        read.assert_not_awaited()
        self.assertNotIn("requestedItems", original)

    async def test_missing_capture_rejects_malformed_or_duplicate_tool_receipt_ids(self):
        journal.current_operation_capture.set(None)
        for rows in ([{**CREATES[0], "id": True}], [CREATES[0], CREATES[0]], [{"id": 1}]):
            with self.subTest(rows=rows):
                result = await journal.invoke_with_operation_journal(
                    "keytao_batch_add_to_draft", {"confirmed": True, "items": CREATES},
                    AsyncMock(return_value={"success": True, "batchId": "s60-delta", "writtenItems": rows}), self.get_tool,
                )
                self.assertEqual(result["writtenItems"], [])
                self.assertTrue(result["receiptItemsUnavailable"])

    async def test_missing_before_snapshot_prevents_write_and_preserves_journal(self):
        result, read, tool = await self.invoke(None, snapshot([OLD, *CREATES]))
        self.assertIs(result["success"], False)
        tool.assert_not_awaited()
        self.assertEqual(read.await_count, 1)
        self.store.save_completed_operation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
