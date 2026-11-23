import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import nonebot

with patch("nonebot.config.dotenv_values", return_value={}):
    nonebot.init()

from keytao_bot.plugins import qq_watchdog as watchdog


class QQWatchdogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        states = patch.multiple(
            watchdog, _reported={}, _pending={}, _checking=set(),
            _background_tasks=set(),
        )
        states.start()
        self.addCleanup(states.stop)

    async def test_reconnect_detects_offline_account_with_cached_login_info(self):
        bot = SimpleNamespace(
            get_login_info=AsyncMock(return_value={"user_id": 123}),
            get_status=AsyncMock(return_value={"online": False, "good": True}),
        )
        with (
            patch.object(watchdog.asyncio, "sleep", new_callable=AsyncMock),
            patch.object(watchdog, "get_bots", return_value={"123": bot}),
            patch.object(watchdog, "_send_tg", new_callable=AsyncMock, return_value=True) as send,
        ):
            await watchdog._verify_login_after_reconnect(bot, "123")

        send.assert_awaited_once()
        self.assertIn("QQ 账号已掉线", send.await_args.args[0])
        bot.get_login_info.assert_not_awaited()

    async def test_heartbeat_retries_delivery_and_resets_dedup_after_recovery(self):
        bot = SimpleNamespace(get_status=AsyncMock(side_effect=[
            {"online": False}, {"online": False}, {"online": False},
            {"online": True}, {"online": False},
        ]))
        with (
            patch.object(watchdog, "QQBot", SimpleNamespace),
            patch.object(watchdog, "get_bots", return_value={"123": bot}),
            patch.object(watchdog.asyncio, "sleep", side_effect=[None] * 5 + [asyncio.CancelledError()]),
            patch.object(watchdog, "_send_tg", side_effect=[False, True, True]) as send,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await watchdog._heartbeat_loop()

        self.assertEqual(bot.get_status.await_count, 5)
        self.assertEqual(send.await_count, 3)
        self.assertTrue(all("QQ 账号已掉线" in sent.args[0] for sent in send.await_args_list))

    async def test_unavailable_status_is_distinct_from_confirmed_offline(self):
        bot = SimpleNamespace(get_status=AsyncMock(side_effect=[
            asyncio.TimeoutError(), {"online": "false"}, {"online": False},
        ]))
        with (
            patch.object(watchdog, "get_bots", return_value={"123": bot}),
            patch.object(watchdog, "_send_tg", return_value=True) as send,
        ):
            for _ in range(3):
                await watchdog._check_qq_status(bot, "123")

        self.assertEqual(send.await_count, 2)
        self.assertIn("QQ 在线状态暂不可确认", send.await_args_list[0].args[0])
        self.assertIn("QQ 账号已掉线", send.await_args_list[1].args[0])

    async def test_transport_alert_waits_for_grace_and_retries(self):
        with (
            patch.object(watchdog.asyncio, "sleep", new_callable=AsyncMock) as sleep,
            patch.object(watchdog, "_send_tg", side_effect=[False, True]) as send,
        ):
            await watchdog._delayed_notify("123", "12:53:11")

        self.assertEqual([call.args for call in sleep.await_args_list], [
            (watchdog.RECONNECT_GRACE,), (watchdog.HEARTBEAT_INTERVAL,),
        ])
        self.assertEqual(send.await_count, 2)
        self.assertTrue(all("QQ Bot 与 NapCat 连接中断" in sent.args[0] for sent in send.await_args_list))

    async def test_only_telegram_acknowledgement_marks_delivery_successful(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.side_effect = [
            httpx.Response(200, json={"ok": False}),
            httpx.Response(429, json={"ok": False}),
            httpx.Response(200, json={"ok": True}),
        ]
        with (
            patch.multiple(watchdog, TELEGRAM_TOKEN="fixture-token", NOTIFY_CHAT_ID="fixture-chat"),
            patch.object(watchdog.httpx, "AsyncClient", return_value=client),
        ):
            self.assertFalse(await watchdog._send_tg("Fixture alert"))
            self.assertFalse(await watchdog._send_tg("Fixture alert"))
            self.assertTrue(await watchdog._send_tg("Fixture alert"))

    async def test_startup_watch_cancels_transport_alert_if_qq_connects_in_grace(self):
        bot = SimpleNamespace(self_id="123", get_status=AsyncMock(return_value={"online": True}))
        with (
            patch.dict(watchdog.os.environ, {"ACCOUNT": "123"}),
            patch.object(watchdog, "QQBot", SimpleNamespace),
            patch.object(watchdog, "get_bots", side_effect=[{}, {"123": bot}, {"123": bot}]),
            patch.object(watchdog, "_heartbeat_loop", side_effect=asyncio.Event().wait),
            patch.object(watchdog.asyncio, "sleep", new_callable=AsyncMock),
            patch.object(watchdog, "_send_tg", return_value=True) as send,
        ):
            try:
                await watchdog.start_heartbeat()
                pending = watchdog._pending["123"]
                before_connect = set(watchdog._background_tasks)
                await watchdog.on_qq_connect(bot)
                await asyncio.gather(*watchdog._background_tasks.difference(before_connect))
                self.assertTrue(pending.cancelled())
            finally:
                await watchdog.stop_heartbeat()

        bot.get_status.assert_awaited_once()
        send.assert_not_awaited()
