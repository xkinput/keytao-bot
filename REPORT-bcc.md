# S63 BCC：独立审查修复报告

已实现 B1–B3 和 SF1–SF6，保留原 S63 的本地导入、统一比较器及只读工具接线。
本轮从已有 S63 脏工作树继续；基线 HEAD 为
`9733c695581b83b391b4980fba33cd607fac3513`。仅本地修改，未提交、推送、部署或访问生产。
六套离线测试、`e2e.test_safety` 及 S63（16 + 8 + 2 = 26 tests）最终全部通过。

## 逐项闭环

| 审查项 | 状态 | 修复与证据 | 回归覆盖 |
|---|---|---|---|
| B1：未收录当零、任意命中获胜 | FIXED | `bcc_reference.py:97` 只接受双方 attested；未命中 `perMillion=null`。不可用时进入旧 jieba/词典分支；旧分支的 attested 只由旧信号计算，避免 BCC 间接改变原规则。见 `keytao_review.py:5782`。 | `test_s63_bcc.py:71,77,90,186`；原错误断言已替换；512 组单边 BCC / 旧信号组合验证 verdict 和 reason 不变；低频双方 6 vs 9 仍为 close。 |
| B2：现代新词 escape hatch 被封死 | FIXED | 删除 `bcc_` reason 早退及 BCC-attested 的 dictionary-dominated 早退；恢复原有候选审查与两个排序方向的语义 override。见 `keytao_review.py:6255,6412`。 | `test_s63_bcc.py:118`：真实 BCC 命中 + 词典收录夹具，直接 override 和正反两个 chain 入口均可到达；只使用既有语义结果 fake，无模型请求。 |
| B3：损坏旧 DB 使启动失败 | FIXED | `bcc_reference.py:34` 整个复制路径捕获 SQLite 错误；SAVEPOINT 防止只复制一半 BCC 数据，失败告警并返回基础构建。 | `test_s63_bcc_ingest.py:122`：旧文件为非 SQLite 字节，实际 builder 重建成功，基础词频可读且 quick_check=ok；原保留测试亦通过。 |
| SF1：安装 BCC 后网页回退死路 | FIXED | `keytao_review.py:5688,5982` 使用 attested，不使用仅表示表安装状态的 available。双方都无本地信号时恢复既有网页回退；单词查询同理。 | `test_s63_bcc.py:52` 验证真实 BCC 表存在、查询词未收录时到达 fake 回退。状态机不再使用不存在的 DB，详见下节。 |
| SF2：字/词分母不同 | FIXED | 跨类型用各自发布表内的 competition rank / row_count，禁止直接比字表与词表 ppm。见 `bcc_reference.py:62,97`、`ingest_bcc.py:130`。 | `test_s63_bcc_ingest.py:40`：字的 ppm > 词的 2 倍，但相对排名 1/2 vs 1/10，词获胜；反向一致；缺排名时回退。 |
| SF3：口语权重偏低 | FIXED | 同类型 headline 改为四频道 max ppm，不再使用 0.7/0.1/0.1/0.1；只有最高信号完全相同且双方都有多领域证据才用多领域破平。 | `test_s63_bcc.py:194`：口语独有与多领域独有的相同 ppm 得到 close，口语不打折。 |
| SF4：磁盘仅采样、没有 gate | FIXED | `ingest_bcc.py:140,187` 在创建 staging、下载 ZIP 前检查真实 free space；同设备合并预算，空间不足拒绝。下载清单前也检查临时目录。 | `test_s63_bcc_ingest.py:63`：free=1 时 download/staging 均零调用，原 DB 字节不变。 |
| SF5：冗余/反向佐证文案 | FIXED | `bcc_reference.py:127` 省略未命中频道、集中说明缺失；`keytao_review.py:5728` BCC 判定只展示 BCC，词典判定不展示反向 jieba；未知结论只列实际 BCC 证据。 | `test_s63_bcc.py:174`：反向 jieba 大值不出现在 BCC 判词中；无双空频道；`test_s63_bcc_delivery.py` 覆盖最终只读回复。 |
| SF6：启动未接入导入 | FIXED | `Dockerfile:15` 改为基础 builder → BCC ingest → bot；BCC ingest 的非零退出通过 warning 分支降级，无需手工进入容器导入。 | `test_s63_bcc_ingest.py:77,101`：读取真实 Docker CMD，替身 uv 对 ingest 返回 0/1/2/137 后均到达 bot；真实 CLI 的超时/坏缓存返回失败且不发布。 |

表中省略前缀：`bcc_reference.py`、`keytao_review.py` 位于 `keytao_bot/utils/`，
`ingest_bcc.py` 位于 `scripts/`。所有 blockers / should-fixes 均已采纳，无 REBUTTED 或 SKIPPED 项。

附带 NIT 处理：

- 不再要求输入按计数降序；真实 rank 由 SQL `RANK() OVER (ORDER BY raw_count DESC)` 计算，同频并列。
  跨类型回归特意采用乱序输入。
- `ingest_bcc.py:217` 明确发布依赖单写者；stat 检查与 replace 之间仍有 TOCTOU 窗口，
  不是跨进程 compare-and-swap，也不宣称掉电安全。
- `preserve_bcc` 使用显式列名复制，兼容旧快照缺少新增排名列。
- `COMMONNESS_BCC_SCORE_SATURATION_PER_MILLION=20` 已命名；
  0.75/0.25 仅构造 score / scoreDelta，不决定本地比较 verdict，不宣称为判定投票权重。
- 补充独立只读审查发现跨字词平局破除后摘要仍展示原最高排名。已增加
  `_balanced_tiebreak` reason，显示实际采用的多领域排名；
  `test_s63_bcc.py:106` 覆盖“最高同为前 10%，多领域前 10% vs 前 50%”。
  复核正反向均通过。该轮未作为跨模型验证宣称。

## 新决策规则

1. 长度为 1 使用 char 表，其他使用 word 表；仅四个现代频道齐备才启用 BCC。
   缺失计数、频次、排名均保持 null；没有用零填补未收录。
2. BCC 只有双方 attested 才可决定方向：
   - 同类型：各取四频道最高 ppm，沿用 2.0 比值门槛；不足两倍为 close。
   - 跨类型：每频道在**完整的自身表**中计算
     `frequency_rank = 1 + count(rows with strictly greater raw_count)`，
     `rankFraction = frequency_rank / row_count`。各取最靠前（最小）的相对排名，
     比较其倒数，仍用 2.0 门槛。相对排名衡量表内位置，不是绝对使用概率。
   - 主信号完全相同、且双方多领域信号都存在时，用多领域同单位信号破平；
     只有一方多领域命中不据此决定。破平不足两倍仍为 close。
3. 单边 attested，或跨类型排名不完整：BCC 不作方向结论，原 jieba 比值、
   单边语料+词典条件、词典 presence margin 保持不变。rank 列缺失的旧 BCC 库
   仍可做同类型比较；不会跨字/词硬比 ppm。
4. 现代语义 override 的资格与旧实现一致：要求已有高置信度语义审查，
   对方是原规则定义的 dictionary-dominated。BCC 命中不会封死它；
   纯只读常用度工具没有额外语义审查/模型调用。
5. 双方所有本地信号均无时，异步比较器恢复既有网页回退。
   `lookup_word_commonness` 只读工具仍纯离线，未知就返回不足。
   运行时 BCC 查询模块不会下载数据。
6. 原始次数、每表分母、完整排名、完整行数、更新时间、TXT/ZIP 哈希都保留。
   fixture 从完整数据库提取排名，不能拿 18 行切片重新计算分母或排名。

## 五组指定重放

全部为离线重放，没有访问生产服务器。前三组读取本地完整数据库；
后两组保留真实 BCC，按审查复现条件注入旧信号：户晨风 dictionary presence=2，
沃集鲜 jieba=500 / dictionary presence=0。没有把注入值写回完整数据库或真实 fixture。

| 比较（左 vs 右） | 修复后 verdict | reason | 交换方向 |
|---|---|---|---|
| 情报所 / 敲不死 | not_enough_evidence | local_signal_insufficient | not_enough_evidence |
| 环境法 / 户晨风 | not_enough_evidence | local_signal_insufficient | not_enough_evidence |
| 五角星 / 沃集鲜 | front_more_common | corpus_and_dictionary_vs_absent | behind_more_common |
| 户晨风（词典 2）/ 一一化（BCC 6） | front_more_common | dictionary_presence_margin | behind_more_common |
| 沃集鲜（jieba 500）/ 一一化（BCC 6） | not_enough_evidence | local_signal_insufficient | not_enough_evidence |

前两组回到旧证据不足是 B1 产品决策的必要结果；不再把单边 BCC 观测当作新词罕见的证据。
沃集鲜无需被武断判成更常用，但不会输给“一一化”的 6 次 BCC 观测；
“一一化”反向作为新词，也不能凭这个命中获得挤占资格。

最终比较摘要示例：

```text
常用度信号不足：「情报所」BCC 多领域 34（每百万 0.08）；新闻 99（每百万 0.11）；「敲不死」BCC 四个现代频道均未收录（发布表最低计数为 6，未收录不等于零次）；单边 BCC 收录不决定高低
「户晨风」较「一一化」更常用：词典收录 2 vs 0；单边 BCC 收录不决定高低
```

保留审查建议的短证据格式，但不会在修复后无足够证据时仍声称“情报所更常用”。
完整结果及输入证据：`/tmp/keytao-s63-bcc/review-pairs.json`；
重放入口：`/tmp/keytao-s63-bcc/review_pairs.py`。

## 真实缓存、排名与资源

本轮无网络请求、无 BCC 下载。沿用此前已缓存的八份现代发布表，
快照时间均为 `2026-05-22T02:53:18Z`。数据来源是
[BCC 静态下载清单](https://bcc.blcu.edu.cn/api/datasets)；
近代汉语、古代汉语没有进入查询或本轮导入。

| 频道 | 字表行数 / 总计数 | 词表行数 / 总计数 |
|---|---:|---:|
| 多领域 | 7,898 / 644,366,519 | 448,679 / 403,819,602 |
| 新闻 | 8,360 / 1,671,404,370 | 786,556 / 928,179,976 |
| 文学 | 7,059 / 160,173,470 | 166,460 / 105,920,737 |
| 口语 | 5,823 / 209,814,959 | 136,405 / 147,994,329 |

八个缓存文件名为 `multi_domain_total_{char,word}_freq.txt.zip`、
`news_total_{char,word}_freq.txt.zip`、`literature_{char,word}_freq.txt.zip`、
`dialogue_{char,word}_freq.txt.zip`。完整 TXT 合计 20,800,536 字节；
本地发布表最低计数为 6，不能由未收录推导真实零次。它也是旧快照，不代表之后的新词使用情况。

本轮实际命令（经离线 guard 启动）：

```sh
.venv/bin/python scripts/ingest_bcc.py --inventory /tmp/keytao-s63-bcc/inventory.json --cached-only
.venv/bin/python scripts/extract_bcc_fixture.py --db data/pinyin_reference.db --output e2e/fixtures/bcc/s63.json
```

`--cached-only` 必须显式提供已下载清单；只读缓存 ZIP，校验成员、CRC、内容大小、
正整数计数与唯一 token，不调用下载函数。此次对 1,567,240 行重导入并保存完整排名：

| 指标 | 本轮实测 |
|---|---:|
| 更新数据集 / 下载 | 8 / 0 |
| 导入前 DB | 157,868,032 字节 |
| 导入后 DB | 162,754,560 字节 |
| 排名升级增加 | 4,886,528 字节 |
| 缓存 | 9,381,717 字节 |
| 导入耗时 | 12.925 秒 |
| 峰值 RSS | 89,915,392 字节（85.750 MiB） |
| 磁盘峰值采样 | 421,544,349 字节（402.016 MiB） |

测量文件：`review-cache-ingest.json`、`review-cache-ingest.stderr`（均在上述临时目录）。
原报告的约 69 MiB RSS 属于旧导入器，不再作为含排名升级的本轮数字。
采样每 20 ms；没有把采样最大值当硬上界或生产容量证明。

磁盘 gate 按设备合并预算：基础 DB 所在设备预留
`2 × old_db_size + 12 × changed_txt_size + 64 MiB`，
缓存设备额外预留 `changed_file_count × 64 MiB`，每个设备再保留 256 MiB。
前者覆盖临时 DB、journal、排名排序与下载暂存的保守空间预算；后者覆盖缓存增长。
不足则在 staging/ZIP 下载前拒绝。它不是磁盘配额或跨进程预留锁。

仍采用 SQLite backup → 同目录 staging → 整批提交 → quick_check → 原子 replace。
WAL 守卫及发布前 inode/size/mtime 校验保留。失败不发布部分 DB；
缓存文件可以先更新，因此不能将缓存更新等同于数据库发布。
排名列在临时库升级，旧库在发布前保持可读。重建基础库时 BCC 保留失败可丢弃可选快照，
不会覆盖“基础库能自愈并启动”的优先级。

幂等复验通过：下载函数被设为失败桩、实际调用 0 次，updated/downloaded 均为空，
DB SHA-256 前后均为 `dd608eea8e2737e6c0e2c2c1bf09a0b6ad633ac269374c60b93524801fe352d3`。
完整库 `quick_check=ok`、缺失排名 0；18 条 fixture 排名逐条用完整表重算核对通过。
再次提取 fixture 后 `cmp` 一致；当前真实切片为 7,932 字节。
证据在 `/tmp/keytao-s63-bcc/verified-artifacts.json`，含本轮所有 Python 改动文件、Dockerfile、
fixture 的 SHA-256。13 个 Python 文件通过 `compile()` 语法检查，`git diff --check` 通过。
数据许可沿用已有 S63 来源记录：官方帮助允许免费下载使用、研究使用需规范引用 BCC 论文；
未擅自标注 CC0/MIT，本轮未联网重新核验许可或论文信息。

## 部署接线与边界

**旧实现只运行 builder && bot，确实需要手工导入，部署会使 BCC 功能处于未安装状态。**
本轮已在 `Dockerfile:15` 接入自动导入：

```sh
uv run python scripts/build_pinyin_reference.py && { uv run python scripts/ingest_bcc.py --timeout 15 || printf '%s\n' 'WARNING: BCC ingest failed; starting bot with existing local signals' >&2; } && exec uv run python bot.py
```

- builder 完成后，每次容器启动检查官方静态清单；有变化才导入，正常幂等时不下载 ZIP。
  无需人工进入容器执行 BCC 命令。部署环境此默认路径可能联网；**本轮没有执行它联网**。
- 每个请求使用 15 秒限制，无自动循环重试。BCC 网络异常、JSON/ZIP/CSV 错误、磁盘不足、
  权限或导入进程非零退出均落入 warning 分支，然后执行 bot；原基础参考库仍可用。
- 已安装可用 BCC 时，导入失败保留旧快照；没有 BCC 时继续使用原 jieba/词典信号。
  BCC optional schema 的保留错误也不会中断基础 builder。
- shell 回归只证明真实 CMD 的控制流，uv/bot 是无网络替身；
  builder 损坏库恢复和 importer 超时/坏缓存另有真实本地执行。
  **没有 build/run Docker、SSH、服务重启、远端 revision/health 检查或生产行为验证。**
- 仍须单写者启动：共享同一 DB 的多个容器不应同时导入。磁盘 gate 不代替共享宿主机容量评估；
  没有在生产增加容器、worker、浏览器会话或任何常驻负载。

## 离线验证记录

补充 `test_s63_bcc.py:212` 直接覆盖未收录现代词原先会输给有词典/BCC 证据占位词的场景：
“沃集鲜”无 BCC 命中、对方“一一化”词典 presence=2 且 BCC=6，原比较先判后者胜出；
已有高置信度语义审查可在两个 chain 方向恢复“沃集鲜”优先，模型调用为 0。

所有应用侧验证使用隔离环境、阻断 socket connect/DNS/sendto，且阻止读取
真实 `.env*` / `.e2e_key`。fake dotenv 不加载生产配置。
入口 `/tmp/keytao-s63-bcc/run_offline.py`；
S63 的 `e2e.s63` 自带子进程隔离，完全独立于 provider rig。

首个红回归命令为：

```sh
.venv/bin/python -m unittest test_s63_bcc test_s63_bcc_ingest
```

`review-red.log` 记录 `Ran 17 tests` / `FAILED (failures=13, errors=1)`：
包括 B1 两个确切误判、B2 override 早退、B3 `file is not a database`，
以及网页回退、口语权重和旧 S63 断言失败。随后修复后 S63 转绿。

状态机首轮恢复真实 DB 后的唯一失败是
`test_word_commonness_short_circuits_accepted_entity` /
`short-circuit keeps entity knowledge`（2022/2023）。
实库复查：“敬德”BCC 多领域 41、新闻 74、文学 10，实际 attested=true；
因此 SF1 的 attested guard 不应为它触发网页回退。
测试现改用实库未收录的 `未收录实体测试词`，并断言未收录，保留全部原实体 mock/成功与短路检查。
不再替换 DB 路径，不删除实库数据，也没有为了测试改生产判定。
`review-state-real-db.log` 已得到 `Results: 2023/2023 passed, 0 failed`。

最终统一重跑由 `/tmp/keytao-s63-bcc/run_suites.py verified-` 调度，
内部用 `run_offline.py` 隔离执行以下命令；8 个指定入口均 exit 0，调度器 exit 0。
完整执行清单为 `/tmp/keytao-s63-bcc/verified-suites.json`，日志为同目录 `verified-*.log`。
以下计数及耗时直接摘自最终日志尾部；没有把前一轮 2022/2023 写成通过。

```text
$ .venv/bin/python test_state_machine.py
============================================================
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED
============================================================

$ .venv/bin/python test_memory_safety.py
----------------------------------------------------------------------
Ran 407 tests in 197.063s

OK

$ .venv/bin/python test_security_fixes.py
============================================================
Results: 268/268 passed, 0 failed
============================================================

$ .venv/bin/python test_review_gate.py
============================================================
Results: 443/443 passed
✅ ALL TESTS PASSED

$ .venv/bin/python test_llm_policy.py
----------------------------------------------------------------------
Ran 11 tests in 0.102s

OK

$ .venv/bin/python test_word_discovery.py
============================================================
Results: 290/290 passed
✅ ALL TESTS PASSED

$ .venv/bin/python -m e2e.test_safety
----------------------------------------------------------------------
Ran 107 tests in 0.677s

OK
```

`.venv/bin/python -m e2e.s63` 各子进程的尾部及最终汇总原文：

```text
Ran 16 tests in 0.147s

OK

Ran 8 tests in 0.607s

OK

Ran 2 tests in 0.029s

OK
{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": true, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 0}, {"module": "test_s63_bcc_delivery", "exit": 0}]}
```

同轮额外相关回归全部 OK：`test_s59_commonness_evidence` 9、`test_s59_commonness_route` 9、
`test_s59_tool_prompt` 6、`test_s56_commonness` 10、`test_sep19_commonness` 6。
四个审查测试缺口均已覆盖：未收录侧具有词典证据、语义 override 两个入口、损坏旧数据库自愈、
字/词不同分母且 ppm 与表内排名方向相反。最后 HEAD 仍为 `9733c69`，全部改动保持未提交。

**ZERO paid model calls / ZERO real-provider calls / ZERO BCC downloads（本轮）。**
没有 `e2e.run`、`pnpm test`、生产密钥读取、生产访问、commit、push 或 deploy。
