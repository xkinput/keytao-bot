"""
Date Time Skill Tools
日期时间工具实现
"""
from datetime import datetime
from typing import Dict, Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception


WEEKDAY_LABELS = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日",
}


async def get_current_datetime(timezone: Optional[str] = None) -> Dict:
    """
    Get current date and time
    获取当前日期和时间

    Args:
        timezone: Optional IANA timezone, such as Asia/Shanghai

    Returns:
        dict: Structured datetime result
    """
    try:
        if timezone:
            if ZoneInfo is None:
                return {
                    "success": False,
                    "error": "当前运行环境不支持时区查询"
                }

            try:
                now = datetime.now(ZoneInfo(timezone))
                timezone_name = timezone
            except ZoneInfoNotFoundError:
                return {
                    "success": False,
                    "error": f"无效时区: {timezone}"
                }
        else:
            now = datetime.now().astimezone()
            timezone_name = str(now.tzinfo) if now.tzinfo else "local"

        return {
            "success": True,
            "timezone": timezone_name,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "iso": now.isoformat(),
            "weekday": now.strftime("%A"),
            "weekday_cn": WEEKDAY_LABELS[now.weekday()],
            "timestamp": int(now.timestamp()),
        }
    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "获取当前日期和时间。当用户询问现在几点、今天几号、今天星期几、当前日期时间，或查询某个时区当前时间时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "可选，IANA 时区名称，如 Asia/Shanghai、Asia/Tokyo、UTC"
                    }
                },
                "required": []
            }
        }
    }
]


TOOL_FUNCTIONS = {
    "get_current_datetime": get_current_datetime
}