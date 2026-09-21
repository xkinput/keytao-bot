# S63 BCC 真实数据切片

`s63.json` 来自 2026-09-21 下载的 BCC 官方静态数据集，所有文件的
`updated_at` 为 `2026-05-22T02:53:18Z`。只包含 18 条命中记录、11 个查询词、
8 份完整表的元数据，以及当前仓库 jieba/词典表中这些词的记录。
未命中的词保留在 `tokens` 中；没有虚构计数。完整词频表和数据库均在 Git 忽略的
`data/` 下，不随代码提交。

生成方式（先导入完整本地数据，再按精确词项筛选）：

```sh
.venv/bin/python scripts/build_pinyin_reference.py
.venv/bin/python scripts/ingest_bcc.py
.venv/bin/python scripts/extract_bcc_fixture.py --db data/pinyin_reference.db --output e2e/fixtures/bcc/s63.json
.venv/bin/python -m e2e.s63
```

提取脚本从完整数据库保留每份文件的总计数、完整行数、更新时间、TXT SHA-256、
ZIP SHA-256 和每个 token 的 `frequency_rank`。该排名按原完整表的计数降序计算，
同频并列，不能用这 18 行的小计重新计算每百万频次或排名。
`的` 同时命中字表和词表，用于检验单字确实采用字频；`龘` 仅命中新闻字表，
`一一化` 在多领域词表有 6 次，用于检验辅助频道、低频观测以及“单边命中不能决定高低”。
两个未知造词在全部选定文件中均无匹配。

审查修复的重导入仅使用 `data/bcc` 缓存和此前保存的官方清单，命令为：

```sh
.venv/bin/python scripts/ingest_bcc.py --inventory /tmp/keytao-s63-bcc/inventory.json --cached-only
```

注入的词典 presence=2、jieba=500 仅存在于审查回归用例中，未改写此真实切片。

来源：[BCC 下载清单](https://bcc.blcu.edu.cn/api/datasets)，下载 URL 为
`https://bcc.blcu.edu.cn/api/datasets/<filename>/download`。每份 ZIP 只含同名 TXT，
CSV 表头为 `token,count`。实测各文件的最低计数均为 6，故“未收录”不能理解为实际零次。

许可说明依据本轮任务提供的 [BCC 官方帮助说明](https://bcc.blcu.edu.cn/help.html)：
“下载中心当前提供 BCC 在线语料库的字词频统计数据，均可免费下载使用；如在研究中使用相关数据，
请规范引用 BCC 论文。”这是官方使用说明，不应擅自标成 CC0 或其他开放许可证。
研究使用需要按官方要求引用 BCC 论文；本轮网络权限仅覆盖数据集下载，未另行核验论文书目信息。
