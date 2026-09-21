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

## S63 follow-up：古代汉语与近代汉语（2026-09-22）

本节是对上文八份现代数据结论的追加。开始时 `git log -1` 核实
HEAD 为 `67ddb7a3124d1115aaf143cc50fa6df5da7ca4e2`，工作区干净。
本次只在本地实施和验证；未提交、推送、部署或访问生产服务器。

### 十二份数据与使用边界

沿用同一 `bcc_frequency` / `bcc_dataset` 管线及独立 dataset 标签，已加入：

| 文件 | TXT 字节 | 新增行 | 总计数 |
|---|---:|---:|---:|
| `classical_chinese_char_freq.txt` | 169,256 | 20,138 | 1,432,519,927 |
| `classical_chinese_word_freq.txt` | 169,256 | 20,138 | 1,432,519,927 |
| `modern_chinese_char_freq.txt` | 77,839 | 8,690 | 1,434,078,685 |
| `modern_chinese_word_freq.txt` | 15,698,705 | 1,377,730 | 835,646,749 |

官方清单仍标为 `2026-05-22T02:53:18Z`。两份古代汉语 TXT 内容 SHA-256
完全相同（`d2b9a0e984b37754e193c0aac168b2eda443a8307a919e4bf1f880e75d16b8f3`），
这是本次实际下载内容的事实；分别按官方 char / word 标签保留，没有合并、修正或编造词条。
全部 12 个数据集共 **2,993,936 行**，比原 1,567,240 行增加 **1,426,696 行**。

- `bcc.channels`、`perMillion`、`rankFraction`、`attested` 仍只由多领域/新闻/文学/口语决定。
  headline 仍为现代四频道最大 ppm；历史数据不参与 score，也不改变现代未收录的 null。
- 两个历史频道单列 `bcc.historicalChannels`，每个字段明确标注频道、数据集、原始次数、ppm、
  自身表内排名/行数、分母、快照时间与哈希。未收录为 null，未安装为 `available=false`。
- 仅双方现代四频道完整可用且双方均未收录、jieba 比值与词典证据没有方向结论时，
  历史频道才可破平。原比较为 `close` 或 `not_enough_evidence` 才进入；词典 presence
  差达到原门槛时仍禁止历史破平。双方须在**同一个历史频道**均收录，不跨时期拼接信号。
  双方都有可比近代汉语时优先近代；否则尝试古代。近代可比但未达到两倍门槛时保留原结论，
  不再挑古代寻找胜方。同类型沿用 2.0 比值门槛，字词混比使用各自完整表的 rank / row_count，
  排名缺失则不硬比 ppm。
- 单边现代收录仍不能由 BCC 单独判定高低；历史高频不能获得针对现代收录词的短码优先权。
  原 jieba / 词典回退与现代新词语义 override 保留，未知新词的网页回退资格不变。
- 只读 `keytao_word_commonness` 始终返回两个历史字段，词频排序固定列出「历史补充」，
  包括未参与判定、未收录和未安装的状态。工具描述与 lookup 技能文案同步这项边界。
  审词比较摘要和逐词审词证据仅在历史决定结果或现代无收录时显示历史数字。
  排序序号仍是展示位置，接近/未知关系不表示严格先后。

### 实测导入、幂等与磁盘 gate

只访问了一次官方 `/api/datasets` GET 和四个新文件的
`/api/datasets/<file>/download` GET；没有访问 `/api/freq`、`/api/search` 或其他网络端点。
下载入口禁止重定向，仍按块处理 ZIP/CSV。首次隔离下载包装器因 audit 参数下标错误，
在发出文件请求前退出，原 DB 未改变；修正包装器后完成以下测量。

| 指标 | 本次实测 |
|---|---:|
| 更新数据集 / 下载 ZIP | 4 / 4 |
| 导入前 DB | 162,754,560 字节（155.215 MiB） |
| 导入后 DB | 250,372,096 字节（238.773 MiB） |
| DB 增长 | 87,617,536 字节（83.559 MiB） |
| 新增 ZIP 缓存 | 7,088,726 字节 |
| 总 ZIP 缓存 | 16,470,443 字节 |
| 导入耗时 | 13.963 秒 |
| 峰值 RSS | 83,214,336 字节（79.359 MiB） |
| 磁盘峰值采样 | 429,680,555 字节（409.775 MiB） |
| 本地 gate 复验时可用磁盘 | 24,821,760,000 字节 |

仍采用 SQLite backup → 同目录 staging → 整批提交 → quick_check → 原子 replace；
WAL 和 inode/size/mtime 发布守卫、单写者限制、坏 ZIP 不发布、重建基础库保留 BCC、
低磁盘提前拒绝及 Docker 启动非致命降级均保留并运行回归。

在 DB/缓存同设备情况下，原 gate 公式不变：
`2 × db_size + 12 × changed_txt_size + (changed_file_count + 1) × 64 MiB + 256 MiB`。
本轮四文件增量需要 **1,122,869,568 字节**可用空间；按扩大的 DB 重新导入全部十二份，
保守预算为 **2,084,581,984 字节**，本地实际 gate 通过。
以用户给定生产余量 **8.3 GB（保守按 8,300,000,000 字节）**注入同一 gate，
十二份全量刷新也通过，超出预算约 **6.215 GB**。
若其他占用不变，扣除 DB 和缓存新增量后的生产余量估算约 **8.205 GB**。
这只是依据给定生产快照的容量计算，**没有联网复测生产磁盘、RAM、swap 或容器数**；
16 GB RAM / 无 swap / 约 40 容器同样来自任务提供值。本次不增加任何生产常驻负载。
20 ms 采样峰值不是硬上界；gate 不是磁盘配额，也不解决并发写者竞态。

离线幂等复验 updated/downloaded 均为空，下载函数调用 0 次，DB SHA-256 前后相同：
`f8560eb35ca21080e21c868ea39625b14496cc7be134e91bac05bc1a4ec84ac4`。
完整 DB `quick_check=ok`，缺失排名 0；25 条真实 fixture 的排名均以完整表重新计算核对。
重新提取的 fixture 包含全部 12 份元数据，保留完整表分母与排名。

测量与复验文件均在 `/tmp/keytao-s63-bcc/`：`followup-inventory.json`、
`followup-ingest.json`、`followup-ingest.stderr`、`followup-data-verification.json`。
其中文本日志可能包含应用导入日志，JSON 主体从首个 `{` 开始读取。

### 破平文案与验证范围

以下来自完整真实数据库的离线只读比较；两词均不在现代四频道和基础 jieba / 词典表中：

```text
「元五」较「于前」更常用：现代四频道均未收录，词典与 jieba 无明确方向，按近代汉语频次：「元五」190,661（每百万 228.16） vs 「于前」40,235（每百万 48.15）；仅作历史语料末级破平
```

古代频道分支另用明确的合成 fixture 验证（不是官方语料事实）：

```text
「古例甲」较「古例乙」更常用：现代四频道均未收录，词典与 jieba 无明确方向，按古代汉语频次：「古例甲」1,234（每百万 123400） vs 「古例乙」12（每百万 1200）；仅作历史语料末级破平
```

新增回归涵盖历史高频不能赢现代收录词的短码、正反方向、jieba/词典方向保护、现代 close
不可被历史打破、无现代收录且旧信号 close 时可破平、单边历史缺失、现代数据集不完整、
跨字词按自身排名、缺排名回退、近代优先、注册工具字段、最终排序路由文案、审词显示边界、
十二份原子导入、幂等性和 WAL 拒绝。原 S63 语义 override 等断言继续保留。

主要代码证据：`scripts/ingest_bcc.py:27` 扩展白名单；
`keytao_bot/utils/bcc_reference.py:63` 隔离查询及现代聚合、`:130` 历史频道比较；
`keytao_bot/utils/keytao_review.py:5871` 旧信号之后的末级 gate、`:5757` 明示历史依据的文案；
`keytao_bot/utils/word_commonness.py:117` 补充工具证据、
`keytao_bot/utils/commonness_query.py:75` 排序历史列。

### 最终离线套件原始尾部

沿用已检查的 `run_offline.py` / `guard/sitecustomize.py`：清空非必要环境变量，
禁止真实 socket connect / DNS / sendto、禁止读取生产 `.env*` / `.e2e_key`，
dotenv 使用离线替身。入口 `run_suites.py followup-final-`，清单为
`/tmp/keytao-s63-bcc/followup-final-suites.json`，13 个入口全部 exit 0，调度器 exit 0。
工具说明更新后另重跑状态机，通过结果覆盖同名最终日志。
以下均直接摘自对应 `followup-final-*.log`，不是预期计数：

```text
$ .venv/bin/python test_state_machine.py
============================================================
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED
============================================================

$ .venv/bin/python test_memory_safety.py
----------------------------------------------------------------------
Ran 407 tests in 193.652s

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
Ran 11 tests in 0.110s

OK

$ .venv/bin/python test_word_discovery.py
============================================================
Results: 290/290 passed
✅ ALL TESTS PASSED

$ .venv/bin/python -m e2e.test_safety
----------------------------------------------------------------------
Ran 107 tests in 0.630s

OK

$ .venv/bin/python -m e2e.s63
Ran 22 tests in 0.170s

OK

Ran 10 tests in 1.283s

OK

Ran 3 tests in 0.034s

OK
{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": true, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 0}, {"module": "test_s63_bcc_delivery", "exit": 0}]}

$ .venv/bin/python -m unittest test_s63_fresh_selection test_s63_general_reply
----------------------------------------------------------------------
Ran 25 tests in 0.146s

OK
```

相关回归 `test_s59_commonness_evidence`（9）、`test_s59_commonness_route`（9）、
`test_s59_tool_prompt`（6）、`test_s56_commonness`（10）、`test_sep19_commonness`（6）
也均通过。9 个变更 Python 文件经 `compile()` 检查，`git diff --check` 通过；
再次提取 fixture 与工作区的真实切片 `cmp` 一致。
首轮红回归分别确认旧实现只导入 8 份、缺少历史字段、历史破平落入网页回退、
排序输出没有历史列；对应日志为 `followup-*-red.log`。

**ZERO paid model calls / ZERO real-provider calls。** 网络仅有上文官方静态数据 GET；
未运行 `pnpm test` 或付费 provider rig。HEAD 保持 `67ddb7a`，代码、测试、fixture 与报告
均只在本地修改，未 stage / commit / push / deploy。完整 DB 和 ZIP 位于被忽略的本地 `data/`，
生产仍未安装本轮四份新增数据；本报告不将离线通过或容量计算视为生产验证。
