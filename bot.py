#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keytao-bot entry point
"""
import nonebot
from nonebot.adapters.qq import Adapter as QQAdapter
from nonebot.adapters.telegram import Adapter as TelegramAdapter

# Initialize NoneBot
nonebot.init()

# Get driver instance
driver = nonebot.get_driver()

# Register adapters
driver.register_adapter(QQAdapter)
driver.register_adapter(TelegramAdapter)

# Load plugins
nonebot.load_from_toml("pyproject.toml")
nonebot.load_plugins("keytao_bot/plugins")

if __name__ == "__main__":
    nonebot.run()
