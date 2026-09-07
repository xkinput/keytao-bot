"""Independent adversarial checks built on the actual completed-undo harness."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_s56_undo as base
from keytao_bot.utils import completed_draft_undo as undo


class CompletedUndoReviewTests(unittest.TestCase):
    setUp = base.CompletedUndoTests.setUp
    tearDown = base.CompletedUndoTests.tearDown
    write = base.CompletedUndoTests.write
    cancel = base.CompletedUndoTests.cancel

    def test_definite_pre_delete_preview_rejection_does_not_lock_actor(self):
        async def run():
            await self.write()
            actual = self.server.get

            def get_tool(name):
                if name == "keytao_batch_remove_draft_items":
                    async def rejected(**_args):
                        return {"success": False, "batchId": self.server.batch_id,
                                "contentVersion": self.server.version, "message": "fixture denied preview"}
                    return rejected
                return actual(name)

            undo.start_operation_turn(self.key)
            reply = await undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), get_tool)
            self.assertNotIn("✅", reply)
            self.assertEqual(6, len(self.server.items))
            self.assertFalse(self.store.actor_has_running_undo("qq", "s56-undo"), reply)
        asyncio.run(run())

    def test_lost_recall_reply_resumes_from_actual_tool_claim_reconciliation(self):
        async def run():
            await self.write(submitted=True)
            actual = self.server.get
            draft_tools = base.harness._draft_tools
            reconcile = False

            def get_tool(name):
                if name != "keytao_recall_batch":
                    return actual(name)
                if reconcile:
                    return draft_tools.keytao_recall_batch

                async def lost_reply(**args):
                    if "expected_content_version" not in args:
                        return await actual(name)(**args)
                    fingerprint = self.store.begin("qq", "s56-undo", "recall", {
                        "batchId": self.server.batch_id,
                        "contentVersion": args["expected_content_version"],
                    })
                    self.assertIsNotNone(fingerprint)
                    await actual(name)(**args)
                    return {"success": False, "uncertain": True, "batchId": self.server.batch_id,
                            "message": "fixture lost recall reply"}
                return lost_reply

            undo.start_operation_turn(self.key)
            reply = await undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), get_tool)
            self.assertNotIn("✅", reply)
            self.assertFalse(self.server.submitted)
            self.assertTrue(self.store.actor_has_running_undo("qq", "s56-undo"))
            reconcile = True

            async def listed(*_identity, **args):
                return {**await actual("keytao_list_draft_items")(**args), "status": "Draft"}

            with (
                patch.object(draft_tools, "_draft_mutation_claims", return_value=self.store),
                patch.object(draft_tools, "get_bot_token", return_value="fixture-token"),
                patch.object(draft_tools, "keytao_list_draft_items", side_effect=listed),
                patch.object(draft_tools.httpx, "AsyncClient", side_effect=AssertionError("unexpected network"), create=True),
            ):
                undo.start_operation_turn(self.key)
                reply = await undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), get_tool)
            self.assertIn("撤销整笔", reply)
            self.assertEqual(self.server.items, [base.row(1, "旧", "jq")])
            self.assertFalse(self.store.actor_has_running_undo("qq", "s56-undo"))
        asyncio.run(run())

    def test_restore_does_not_rebind_original_delete_after_target_changes(self):
        async def run():
            target = {"id": 90, "word": "旧", "code": "jq", "type": "Single", "weight": 10,
                      "remark": None, "status": "Finish", "userId": 4}

            def fingerprint():
                return hashlib.sha256(json.dumps(target, ensure_ascii=False, sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()

            self.server.items = [{**base.row(1, "旧", "jq", "Delete"),
                                  "targetPhraseId": target["id"], "targetFingerprint": fingerprint()}]
            self.server.remove_old_on_shift = True
            await self.write()
            original_restore = undo._restore_request

            async def phrases(_method, _path, **_kwargs):
                self.assertEqual(_path, "/api/phrases/by-word")
                return {"phrases": [{**target, "user": {"id": target["userId"]}}],
                        "pagination": {"totalPages": 1}}

            async def changed_target_restore(identity, batch_id, items, **ticket):
                if ticket.get("confirmed") is not True:
                    target["weight"] = 11
                result = await original_restore(identity, batch_id, items, **ticket)
                if ticket.get("confirmed") is True and result.get("success") is True:
                    for item in self.server.items:
                        if item["id"] >= 100:
                            item["targetPhraseId"] = target["id"]
                            item["targetFingerprint"] = fingerprint()
                return result

            with (patch.object(undo.http_client, "keytao_json", side_effect=phrases),
                  patch.object(undo, "_restore_request", side_effect=changed_target_restore)):
                reply = await self.cancel()
            self.assertNotIn("✅", reply)
            confirmed = [ticket for _items, ticket in self.server.restore_calls if ticket.get("confirmed")]
            self.assertEqual([], confirmed, "undo re-bound a stale Delete onto the changed live phrase")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
