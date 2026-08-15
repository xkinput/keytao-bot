#!/usr/bin/env python3
"""Report web channel configuration; opt into external probes with --live."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path


def _load_web_tools():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # The script's stdout is a machine-readable JSON report. Application
    # startup still emits the module's doctor log in the real bot process.
    try:
        from nonebot.log import logger

        logger.remove()
    except Exception:
        pass
    path = root / "keytao_bot" / "skills" / "web-search" / "tools.py"
    spec = importlib.util.spec_from_file_location("keytao_web_channels_doctor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load web-search tools from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report web channel backends and last-known status without probing by default.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly probe lightweight external endpoints through hardened egress",
    )
    args = parser.parse_args()
    tools = _load_web_tools()
    report = asyncio.run(tools.probe_web_channels()) if args.live else tools.web_channels_doctor()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
