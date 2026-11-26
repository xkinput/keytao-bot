"""Offline regression tests ensuring URLs are never glued to trailing brackets or punctuation."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from keytao_bot.plugins.chat_adapters import _send_telegram_plain_chunks
from keytao_bot.plugins.chat_render import (
    _to_markdownv2,
    render_platform_public_links,
    separate_url_boundaries,
    strip_bare_batch_ids,
)
from keytao_bot.utils.pending_confirmation import (
    UNBOUND_BINDING_PRECHECK_NOTICE,
    advertised_command_suggestions,
)


class UrlBoundarySeparationTests(unittest.IsolatedAsyncioTestCase):
    def test_unbound_binding_notice_has_no_trailing_bracket_or_punctuation(self):
        self.assertIn("/bind", UNBOUND_BINDING_PRECHECK_NOTICE)
        self.assertFalse(advertised_command_suggestions(UNBOUND_BINDING_PRECHECK_NOTICE))
        self.assertTrue(
            UNBOUND_BINDING_PRECHECK_NOTICE.endswith("https://keytao.vercel.app/profile")
        )
        for platform, expected_url in (
            ("telegram", "https://keytao.vercel.app/profile"),
            ("qq", "https://keytao.rea.ink/profile"),
            ("web", "https://keytao.vercel.app/profile"),
        ):
            with self.subTest(platform=platform):
                rendered = render_platform_public_links(UNBOUND_BINDING_PRECHECK_NOTICE, platform)
                self.assertTrue(rendered.endswith(expected_url), rendered)
                self.assertNotIn(expected_url + "）", rendered)
                self.assertNotIn(expected_url + "。", rendered)

    def test_all_adjacent_brackets_quotes_and_punctuation_are_separated_from_urls(self):
        cases = (
            (
                "（请在账号页面生成实际绑定码；绑定说明：https://keytao.vercel.app/profile）。",
                "（请在账号页面生成实际绑定码；绑定说明：https://keytao.vercel.app/profile ）。",
            ),
            (
                "（汉典 https://www.zdic.net/hans/%E6%B5%8B%E8%AF%95；维基百科 https://zh.wikipedia.org/wiki/%E6%B5%8B%E8%AF%95）",
                "（汉典 https://www.zdic.net/hans/%E6%B5%8B%E8%AF%95 ；维基百科 https://zh.wikipedia.org/wiki/%E6%B5%8B%E8%AF%95 ）",
            ),
            (
                "规则文档：https://keytao-docs.vercel.app/a、https://keytao-docs.vercel.app/b",
                "规则文档：https://keytao-docs.vercel.app/a 、https://keytao-docs.vercel.app/b",
            ),
            (
                "详情见(https://example.com/path)和[https://example.com/other]",
                "详情见(https://example.com/path )和[https://example.com/other ]",
            ),
            (
                "查看「https://example.com/a」或【https://example.com/b】或《https://example.com/c》",
                "查看「https://example.com/a 」或【https://example.com/b 】或《https://example.com/c 》",
            ),
            (
                "先访问https://keytao.vercel.app/profile，再生成绑定码。",
                "先访问https://keytao.vercel.app/profile ，再生成绑定码。",
            ),
            (
                "参考词条（https://zh.wikipedia.org/wiki/Python_(语言)）。",
                "参考词条（https://zh.wikipedia.org/wiki/Python_(语言) ）。",
            ),
            (
                "已结束：https://example.com/done.",
                "已结束：https://example.com/done .",
            ),
            (
                "草稿地址：https://keytao.vercel.app/batch/1234",
                "草稿地址：https://keytao.vercel.app/batch/1234",
            ),
            (
                "1. 登录：https://keytao.vercel.app\n2. 打开：https://keytao.vercel.app/profile\n完成",
                "1. 登录：https://keytao.vercel.app\n2. 打开：https://keytao.vercel.app/profile\n完成",
            ),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                once = separate_url_boundaries(raw)
                self.assertEqual(once, expected)
                self.assertEqual(separate_url_boundaries(once), expected)

    def test_render_platform_public_links_rewrites_host_without_swallowing_fullwidth_brackets(self):
        raw = "提示（绑定说明：https://keytao.vercel.app/profile）。"
        qq = render_platform_public_links(raw, "qq")
        tg = render_platform_public_links(raw, "telegram")
        self.assertEqual(qq, "提示（绑定说明：https://keytao.rea.ink/profile ）。")
        self.assertEqual(tg, "提示（绑定说明：https://keytao.vercel.app/profile ）。")

    def test_strip_bare_batch_ids_preserves_batch_url_followed_by_fullwidth_bracket(self):
        batch_id = "bc32871a-1111-2222-3333-444444444444"
        raw = f"详情（https://keytao.vercel.app/batch/{batch_id}），批次 {batch_id}"
        cleaned = strip_bare_batch_ids(raw)
        self.assertIn(f"https://keytao.vercel.app/batch/{batch_id}", cleaned)
        self.assertEqual(cleaned.count(batch_id), 1)

    async def test_telegram_markdownv2_and_plain_chunk_send_separate_url_boundaries(self):
        raw = "说明（https://keytao.vercel.app/profile）。"
        mv2 = _to_markdownv2(raw)
        self.assertIn("https://keytao\\.vercel\\.app/profile ）", mv2)

        bot = SimpleNamespace(send=AsyncMock())
        await _send_telegram_plain_chunks(bot, object(), raw)
        bot.send.assert_awaited_once()
        sent_text = bot.send.await_args.kwargs["message"]
        self.assertEqual(sent_text, "说明（https://keytao.vercel.app/profile ）。")


if __name__ == "__main__":
    unittest.main()
