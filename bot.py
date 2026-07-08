#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
keytao-bot entry point
"""
from pathlib import Path

import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotAdapter
from nonebot.adapters.telegram import Adapter as TelegramAdapter
from nonebot.log import logger

# Initialize NoneBot
nonebot.init()


def configure_persistent_logging() -> None:
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "bot.log"
    logger.add(
        log_path,
        rotation="20 MB",
        retention="14 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.info(f"Persistent file logging enabled: {log_path}")


configure_persistent_logging()

# Get driver instance
driver = nonebot.get_driver()

# Register adapters
driver.register_adapter(OneBotAdapter)
driver.register_adapter(TelegramAdapter)

# Load plugins
nonebot.load_from_toml("pyproject.toml")
nonebot.load_plugins("keytao_bot/plugins")

if __name__ == "__main__":
    nonebot.run()
