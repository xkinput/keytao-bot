import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "keytao_bot" / "utils" / "github_sync_notification.py"
SPEC = importlib.util.spec_from_file_location("github_sync_notification_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_github_sync_notification = MODULE.build_github_sync_notification


class GithubSyncNotificationTest(unittest.TestCase):
    def test_formats_detailed_release_summary(self) -> None:
        result = build_github_sync_notification({
            "prUrl": "https://github.com/xkinput/KeyTao/pull/132",
            "releaseTag": "v1.3.5",
            "releaseUrl": "https://github.com/xkinput/KeyTao/releases/tag/v1.3.5",
            "pendingSyncBatches": 28,
            "syncSummary": {
                "contributors": ["Rea", "GarthTB", "朝歌", "EVO"],
                "totalEntries": 39,
                "stats": [
                    {"type": "词组", "create": 38, "change": 1, "delete": 0},
                ],
            },
        })

        self.assertIn("同步 PR：https://github.com/xkinput/KeyTao/pull/132", result)
        self.assertIn("Release：v1.3.5", result)
        self.assertIn("• 总计 39 条词条", result)
        self.assertIn("• 词组：新增 38，修改 1", result)
        self.assertIn("本次词库贡献者（4 位）", result)
        self.assertIn("Rea、GarthTB、朝歌、EVO", result)
        self.assertIn("感谢以上贡献者", result)
        self.assertIn("本次触发时待同步批次：28 个", result)
        self.assertIn("keytao-v1.3.5.zip", result)

    def test_counts_deletions_in_total_entries(self) -> None:
        result = build_github_sync_notification({
            "prUrl": "https://github.com/xkinput/KeyTao/pull/180",
            "releaseTag": "v1.3.35",
            "syncSummary": {
                "contributors": ["Rea", "GarthTB"],
                "totalEntries": 3,
                "stats": [
                    {"type": "词组", "create": 2, "change": 1, "delete": 0},
                    {"type": "单字", "create": 0, "change": 0, "delete": 1},
                ],
            },
        })

        self.assertIn("• 总计 4 条词条", result)
        self.assertIn("• 词组：新增 2，修改 1", result)
        self.assertIn("• 单字：删除 1", result)
        self.assertIn("本次词库贡献者（2 位）", result)
        self.assertIn("keytao-v1.3.35.zip", result)

    def test_keeps_legacy_response_compatible(self) -> None:
        result = build_github_sync_notification({
            "prUrl": "https://github.com/xkinput/KeyTao/pull/132",
            "releaseTag": "v1.3.5",
            "pendingSyncBatches": 28,
        })

        self.assertIn("Release：v1.3.5", result)
        self.assertNotIn("本次更新：", result)
        self.assertIn("本次触发时待同步批次：28 个", result)


if __name__ == "__main__":
    unittest.main()
