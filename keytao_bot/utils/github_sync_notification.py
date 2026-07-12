"""Format GitHub dictionary release notifications for group chat."""
from typing import Any


def _value(data: dict[str, Any], camel_key: str, snake_key: str) -> Any:
    return data.get(camel_key) if data.get(camel_key) is not None else data.get(snake_key)


def _format_type_stat(stat: dict[str, Any]) -> str:
    type_name = str(stat.get("type") or stat.get("typeName") or "未分类").strip()
    parts = []
    for key, label in (("create", "新增"), ("change", "修改"), ("delete", "删除")):
        try:
            count = int(stat.get(key) or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            parts.append(f"{label} {count}")
    return f"• {type_name}：{'，'.join(parts)}" if parts else ""


def build_github_sync_notification(data: dict[str, Any]) -> str:
    pr_url = _value(data, "prUrl", "pr_url")
    release_url = _value(data, "releaseUrl", "release_url")
    release_tag = _value(data, "releaseTag", "release_tag")
    pending_count = data.get("pendingSyncBatches")
    sync_summary = data.get("syncSummary") or data.get("sync_summary")

    lines = [
        "本喵已完成 GitHub 词库同步并发布。",
        f"同步 PR：{pr_url}",
    ]
    if release_tag or release_url:
        lines.append(f"Release：{release_tag or '已发布'}")
    if release_url:
        lines.append(f"发布地址：{release_url}")

    if isinstance(sync_summary, dict):
        contributors = [
            str(item).strip()
            for item in sync_summary.get("contributors", [])
            if str(item).strip()
        ]
        total_entries = sync_summary.get("totalEntries")
        stats = sync_summary.get("stats", [])

        lines.extend(["", "本次更新："])
        if total_entries is not None:
            lines.append(f"• 总计 {total_entries} 条词条")
        if isinstance(stats, list):
            lines.extend(
                formatted
                for stat in stats
                if isinstance(stat, dict) and (formatted := _format_type_stat(stat))
            )

        if contributors:
            lines.extend([
                "",
                f"本次词库贡献者（{len(contributors)} 位）：",
                "、".join(contributors),
                "感谢以上贡献者对键道词库的完善！",
            ])

    if pending_count is not None:
        lines.extend(["", f"本次触发时待同步批次：{pending_count} 个。"])
    lines.append("请大家检查并及时更新词库。")
    return "\n".join(lines)
