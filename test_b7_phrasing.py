"""Offline B7 recode/delete phrasing and existing-word query regressions."""

import os
import unittest

import test_semantic_direct_exec as direct
import test_semantic_query_route as query


class B7PhrasingRouteTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = direct.SemanticDirectExecTests.asyncTearDown
    row = staticmethod(direct.SemanticDirectExecTests.row)
    result = direct.SemanticDirectExecTests.result
    call = direct.SemanticDirectExecTests.call
    stage = direct.SemanticDirectExecTests.stage
    review_word = direct.SemanticDirectExecTests.review_word
    answer = direct.SemanticDirectExecTests.answer

    async def asyncSetUp(self):
        await direct.SemanticDirectExecTests.asyncSetUp(self)
        self.rows = [self.row(word, code) for word, code in (
            ("我送给", "wsg"), ("纯音乐", "jyy"), ("火腿肠", "htj"),
            ("航天局", "htju"), ("富临", "fjlb"),
        )]

    def assert_recodes(self, response, expected):
        self.assertEqual(len(self.writes), 1, (self.calls, response))
        name, args = self.writes[0]
        self.assertEqual(name, "keytao_batch_add_to_draft")
        self.assertEqual([(item["action"], item["word"], item["code"])
                          for item in args["items"]], expected)
        self.assertIn("已加入草稿", response)
        self.assertIsNone(self.store.get_record(self.author))
        self.review.assert_not_awaited()

    async def test_multiple_glued_recodes_write_one_exact_batch(self):
        handled, ctx = await self.stage(
            "修改编码: 我送给为wsgu， 纯音乐为jyya", "recode",
            ("我送给为", "纯音乐为"), ("wsgu", "jyya"),
        )
        self.assertTrue(handled)
        self.assert_recodes(ctx.response, [
            ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
            ("Delete", "纯音乐", "jyy"), ("Create", "纯音乐", "jyya"),
        ])

    async def test_each_connector_keeps_same_word_recode_route(self):
        for connector in ("为", "是", "改为", "改成"):
            with self.subTest(connector=connector):
                self.writes.clear()
                handled, ctx = await self.stage(
                    f"修改编码: 我送给{connector}wsgu", "recode",
                    (f"我送给{connector}",), ("wsgu",),
                )
                self.assertTrue(handled)
                self.assert_recodes(ctx.response, [
                    ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
                ])

    async def test_multiple_recodes_survive_sealed_confirmation(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "recode"
        handled, ctx = await self.stage(
            "修改编码: 我送给为wsgu，纯音乐为jyya", "recode",
            ("我送给", "纯音乐"), ("wsgu", "jyya"),
        )
        self.assertTrue(handled)
        self.assertIn("回复「确认」执行", ctx.response)
        self.assertEqual(self.writes, [])
        self.assert_recodes(await self.answer("确认"), [
            ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
            ("Delete", "纯音乐", "jyy"), ("Create", "纯音乐", "jyya"),
        ])

    async def test_model_code_order_cannot_swap_literal_targets(self):
        for message, operands, codes in (
            ("修改编码: 我送给为wsgu，纯音乐为jyya", ("我送给", "纯音乐"), ("jyya", "wsgu")),
            ("修改编码: 我送给为wsgu，纯音乐为jyya", ("纯音乐", "我送给"), ("jyya", "wsgu")),
            ("修改编码: 我送给为wsgu，纯音乐为jyya", ("我送给", "纯音乐"), ("wsg", "jyya", "jyy", "wsgu")),
            ("我送给改为wsgu，纯音乐改为jyya", ("我送给", "纯音乐"), ("wsg", "jyya", "jyy", "wsgu")),
            ("把我送给改为wsgu，纯音乐改为jyya", ("我送给", "纯音乐"), ("wsg", "jyya", "jyy", "wsgu")),
        ):
            with self.subTest(operands=operands, codes=codes):
                self.writes.clear()
                handled, ctx = await self.stage(message, "recode", operands, codes)
                self.assertTrue(handled)
                self.assert_recodes(ctx.response, [
                    ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
                    ("Delete", "纯音乐", "jyy"), ("Create", "纯音乐", "jyya"),
                ])

    async def test_real_word_ending_in_connector_is_not_truncated(self):
        self.rows.append(self.row("作为", "zw"))
        handled, ctx = await self.stage("修改编码: 作为wsgu", "recode", ("作为",), ("wsgu",))
        self.assertTrue(handled)
        self.assert_recodes(ctx.response, [
            ("Delete", "作为", "zw"), ("Create", "作为", "wsgu"),
        ])

    async def test_ambiguous_connector_never_changes_either_word(self):
        self.rows.extend((self.row("作为", "zw"), self.row("作", "z")))
        handled, ctx = await self.stage("修改编码: 作为wsgu", "recode", ("作为",), ("wsgu",))
        self.assertTrue(handled)
        self.assertIn("唯一确定", ctx.response)
        self.assertEqual(self.writes, [])

    async def test_connector_with_space_before_code_resolves_literal_word(self):
        handled, ctx = await self.stage(
            "修改编码: 我送给为 wsgu", "recode", ("我送给为",), ("wsgu",),
        )
        self.assertTrue(handled)
        self.assert_recodes(ctx.response, [
            ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
        ])

    async def test_ambiguous_second_source_prevents_all_recodes(self):
        self.rows.append(self.row("纯音乐", "jyya"))
        handled, ctx = await self.stage(
            "修改编码: 我送给为wsgu，纯音乐为jyyb", "recode",
            ("我送给", "纯音乐"), ("wsgu", "jyyb"),
        )
        self.assertTrue(handled)
        self.assertIn("指定旧码", ctx.response)
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.store.get_record(self.author))

    async def test_same_word_delete_add_overrides_wrong_model_op(self):
        for message, op, operands, codes in (
            ('删除"纯音乐"编码，添加"纯音乐"为jyya', "delete", ("纯音乐",), ("jyya",)),
            ('删除"纯音乐"的编码jyy，添加"纯音乐"的编码为jyya', "recode", ("纯音乐", "纯音乐"), ("jyya",)),
        ):
            with self.subTest(message=message):
                self.writes.clear()
                handled, ctx = await self.stage(message, op, operands, codes)
                self.assertTrue(handled)
                self.assert_recodes(ctx.response, [
                    ("Delete", "纯音乐", "jyy"), ("Create", "纯音乐", "jyya"),
                ])

    async def test_wrong_explicit_old_code_never_writes(self):
        handled, ctx = await self.stage(
            '删除"纯音乐"的编码jyyb，添加"纯音乐"的编码为jyya',
            "recode", ("纯音乐", "纯音乐"), ("jyya",),
        )
        self.assertTrue(handled)
        self.assertIn("未写入", ctx.response)
        self.assertEqual(self.writes, [])

    async def test_delete_wrong_code_cannot_seal_or_replace_ticket(self):
        for direct_exec in (True, False):
            with self.subTest(direct_exec=direct_exec):
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "delete" + (",direct_exec" if direct_exec else "")
                pending = self.commands.PendingToolConfirm("keytao_submit_batch", {"batch_id": "offline-batch"})
                self.assertTrue(self.store.set(self.author, pending))
                record = self.store.get_record(self.author)
                handled, ctx = await self.stage("删除纯音乐 jyya", "delete", ("纯音乐",), ("jyya",))
                self.assertTrue(handled)
                self.assertIn("jyy", ctx.response)
                self.assertIn("请确认", ctx.response)
                self.assertNotIn("删词 纯音乐 jyya", ctx.response)
                self.assertIs(self.store.get_record(self.author), record)
                self.assertEqual(self.writes, [])

    async def test_delete_guard_accepts_exact_draft_create(self):
        self.rows = [row for row in self.rows if row["word"] != "纯音乐"]
        self.draft_items = [{
            "id": 7, "word": "纯音乐", "code": "jyya", "type": "Phrase",
            "action": "Create", "weight": 100,
        }]
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "delete"
        handled, ctx = await self.stage("删除纯音乐 jyya", "delete", ("纯音乐",), ("jyya",))
        self.assertTrue(handled)
        self.assertIn("回复「确认」执行", ctx.response)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.store.get(self.author).args["codes"], ["jyya"])


class B7QueryRouteTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = query.SemanticQueryRouteTests.asyncSetUp
    asyncTearDown = query.SemanticQueryRouteTests.asyncTearDown
    lookup = query.SemanticQueryRouteTests.lookup
    context = query.SemanticQueryRouteTests.context
    set_model = query.SemanticQueryRouteTests.set_model
    ticket = query.SemanticQueryRouteTests.ticket
    assert_finished = query.SemanticQueryRouteTests.assert_finished
    assert_preserved = query.SemanticQueryRouteTests.assert_preserved

    async def test_present_words_code_query_never_calls_general_or_changes_ticket(self):
        self.entries.update({"我送给": "wsg", "纯音乐": "jyy"})
        record, original, _prompt = self.ticket(self.author)
        ctx = self.context("查询编码：我送给，纯音乐")
        self.set_model(operands=("我送给", "纯音乐"))
        await self.assert_finished(ctx)
        self.assertIn("wsg", ctx.response)
        self.assertIn("jyy", ctx.response)
        self.assertEqual([call.args[1] for call in self.tool.await_args_list],
                         [{"word": "我送给"}, {"word": "纯音乐"}])
        self.main.assert_not_awaited()
        self.assert_preserved(self.author, record, original)

    async def test_comma_query_without_ticket_uses_dictionary_reply(self):
        self.entries.update({"我送给": "wsg", "纯音乐": "jyy"})
        ctx = self.context("我送给，纯音乐")
        self.set_model(operands=("我送给", "纯音乐"))
        await self.assert_finished(ctx)
        self.assertIn("wsg", ctx.response)
        self.assertIn("jyy", ctx.response)
        self.main.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))


if __name__ == "__main__":
    unittest.main()
