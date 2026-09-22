# S65：单词候选常用度证据交付

已修复单词候选回复丢失 BCC 证据的问题，并统一其他比较摘要出口。比较器的胜负规则、阈值、推荐编码和写入授权规则未改。

起始 HEAD：`27663cc`（`fix: only offer 换码 when the live record can actually bind it`）；分支 `main`，起始工作区干净。仅本地修改，未提交、未推送、未部署。

## 根因与渲染入口

**CONFIRMED**：原 `chat_render.py:1389-1395` 的非前插分支自行拼接“不弱于 / 信号不足”，忽略了比较器的证据摘要。最小离线测试在修改前逐字复现了事故回复，并因缺少 `多领域 34（每百万 0.08）` 失败。

**REBUTTED**：不是这条路径没有获取 BCC 数据，也不是交付过滤器删掉数据。`keytao_review.py:6142` 的 `_candidate_commonness_assessment` 已将比较器的 `summary` 和 `decisionReason` 原样放入 assessment。这个 summary 包含完整的 BCC 频道与数字；`validated_front_insert_recommendation` 也保留它。无需再查 DB，也无需把整个 reference 重复装入持久化状态。S65 测试冻结评估后将 `_query_commonness_reference` 替换为抛错函数，仍能完成所有候选渲染和最终交付。

共享出口为 `keytao_bot/utils/commonness_copy.py:4` 的 `render_commonness_summary`：优先原样交付比较器 summary；只有旧记录没有 summary 时才使用统一的保守文案，不补造数字。`:22` 的 `candidate_commonness_summary_copy` 只追加维持排序、推荐空位等操作说明。证据文案仍由 `keytao_review.py:5734` 的 `_reference_comparison_summary` 和 `:5919` 的 `_web_fallback_summary` 生成。

盘点了以下 **17 个用户可见的摘要出口或组合入口**，均在类级调用链测试清单中：

| 位置（`keytao_bot/` 下） | 现在交付的内容 |
| --- | --- |
| `plugins/chat_render.py:1367` `_format_candidate_ordering_assessment` | 前插使用共享推荐文案；其他 verdict 使用共享证据摘要，再追加空位建议。原两条硬编码结论已移除。 |
| `plugins/chat_render.py:1437` `_format_reviewed_add_prompt` | 每条 candidate assessment 经上面的共同入口展示。 |
| `utils/pending_confirmation.py:1645` `front_insert_recommendation_copy` | 推荐调序命令 + 原样共享依据 + 不调序备选；有、无控制文案两个分支共用。 |
| `utils/pending_confirmation.py:1700` `candidate_commonness_guard_copy` | 候选全部被占用时显示各占位词的共享依据，再说明维持排序和点名顶替。 |
| `utils/pending_confirmation.py:1769` `render_server_backed_single_word_candidates` | 重画单词候选时，非前插分支也保留绑定 assessment 的证据。 |
| `utils/pending_confirmation.py:1896` `render_server_backed_single_word_lookup` | 复用审词正文，或经上面的候选重画入口交付证据。 |
| `utils/pending_confirmation.py:1169` `render_server_backed_batch_candidates` | 有审词正文时复用单词入口；简版批量候选也补上非前插证据。 |
| `utils/pending_confirmation.py:1464` `render_server_backed_batch_lookup` | 经批量候选共享入口，去掉写入控制。 |
| `utils/same_code_reorder.py:77` `_advisory` | 共享比较摘要 + 按用户要求执行；移除自行拼接“更常用 / 未分高低”。 |
| `utils/commonness_query.py:75` `render_commonness_table` | 双词比较摘要走共享出口；表格仍使用 S63 的独立频道数据列。 |
| `plugins/chat_commands.py:655` `_protected_eviction_response` | 从词、占位词、占位编码匹配的 assessment 取依据，再说明保留位置；移除自行归纳“不弱于”。 |
| `plugins/chat_commands.py:3213` `_generate_usage_comparison_note` | 只使用有证据的比较结果，经共享出口组合摘要。 |
| `plugins/chat_commands.py:9682` `_format_code_chain_reorder_confirmation` | 调序确认页的 comparison 摘要统一取共享出口。 |
| `plugins/chat_commands.py:9732` `_reorder_evidence_lines` | 调序的 evidence 列表保留共同来源的比较摘要。 |
| `plugins/chat_commands.py:9788` `_format_reorder_noop` | 无需调整时，经上面的 evidence 入口说明依据。 |
| `utils/keytao_review.py:7047` `_chain_recommendation_text` | 将评估器产生的整链摘要经共享出口交付，再列移动方案。 |
| `utils/keytao_review.py:7928` `build_review_note` | 审词报告中的双词比较、整链优先级摘要均经共享出口。 |

比较器内部的整链置信度、调序间距等诊断仍属于评估结果的生产端，没有改动算法。独立词频表行、语料/词典 evidence 行只展示原始信号，不在消费端重判两词胜负。授权门禁和交付过滤器均未放宽。

文案规则：单边 BCC 收录明确写“不决定高低”，不再把整个情况称为没有信号；双边现代 BCC 显示双方词名、频道、四位小数精度及所用频次比；只列至少一边有计数的频道。历史证据只在历史规则决定结果或双方现代未收录时进入比较摘要，未决定时标注“未参与判定”。双方都没有历史计数时只简述未收录，不逐个输出空频道；缺数据时保持未知。只读表格的历史数据列维持 S63 原有契约。

## 四类文案前后对照

以下为旧 HEAD 函数与当前函数在同一份隔离 fixture 上实际输出的摘要。旧函数通过 `git show 27663cc:...` 读取到 `/tmp/keytao-s65/before-*.py`，只执行相关纯渲染函数；未 checkout 或修改 Git 历史。完整输出：`/tmp/keytao-s65/samples.log`。

### 1. 单边收录：敲不死 / 情报所

使用仓库 `e2e/fixtures/bcc/s63.json` 的真实切片：多领域 34 / 0.08419601186175207，新闻 99 / 0.10666034881148954。未连接生产 DB；数字与用户提供的事故证据一致。verdict 始终是 `not_enough_evidence`，推荐始终是 `qbso`。

```text
BEFORE: 常用度评估：「敲不死」与「情报所」的常用度信号不足，按空位 qbso 推荐
AFTER: 常用度评估：「敲不死」BCC 四个现代频道均未收录；「情报所」BCC 多领域 34（每百万 0.08）；新闻 99（每百万 0.11）；单边 BCC 收录不决定高低；推荐空位 qbso
```

### 2. 双边收录：蛋粉 / 单份

fixture 精确复现用户给定的 `0.0944 / 0.0371 = 2.544474…`。**多领域计数 944 / 371 和分母 100 亿是合成的测试数据，不是声称生产的频道与计数如此。** 正反两个方向都验证：蛋粉胜出，显示双方数字和 `2.54×`。旧前插路径已有证据，但精度只有两位且没有倍数；旧反向空位路径会丢失证据。

```text
BEFORE:
推荐：
- “「蛋粉」占 dffn、「单份」顺延”（蛋粉、单份）
依据：「蛋粉」较「单份」更常用：BCC 多领域 944（每百万 0.09）；「单份」BCC 多领域 371（每百万 0.04）
不重排选 2（dffno）。

AFTER:
推荐：
- “「蛋粉」占 dffn、「单份」顺延”（蛋粉、单份）
依据：「蛋粉」较「单份」更常用：「蛋粉」BCC 多领域 944（每百万 0.0944）；「单份」BCC 多领域 371（每百万 0.0371）；所用频次比 2.54×
不重排选 2（dffno）。
```

### 3. 双方未收录

合成词“空例甲 / 空例乙”通过真实本地比较器构造无收录对象，直接接入候选 assessment 和所有渲染入口。未触发另外的联网回退分支。

```text
BEFORE: 常用度评估：「空例甲」与「空例乙」的常用度信号不足，按空位 qbso 推荐
AFTER: 常用度评估：常用度信号不足：「空例甲」BCC 四个现代频道均未收录；「空例乙」BCC 四个现代频道均未收录；历史补充：「空例甲」历史频道均未收录；「空例乙」历史频道均未收录（未参与判定）；推荐空位 qbso
```

### 4. 历史补充决定结果

合成历史计数“古例甲”1234 / “古例乙”12，分母 10000；现代四频道双方未收录。这里用反向候选验证原来“不弱于”分支也交付历史依据。另有正向历史前插及历史只命中一边、不得判胜的测试。

```text
BEFORE: 常用度评估：「古例甲」不弱于「古例乙」，维持现有排序，推荐空位 qbso
AFTER: 常用度评估：「古例甲」较「古例乙」更常用：现代四频道均未收录，词典与 jieba 无明确方向，按古代汉语频次：「古例甲」1,234（每百万 123400） vs 「古例乙」12（每百万 1200）；仅作历史语料末级破平，维持现有排序，推荐空位 qbso
```

## 验证与原始 tails

新增 `test_s65_commonness_render.py`，共 10 个测试方法（包含子场景）：四类要求、双方收录的正反向、接近时不造胜者、历史仅单边收录不判胜、现代命中时隐藏历史比较补充、无重查、单词/批量/保护提示/审词报告及最终门禁交付。

类级 guard 包括：17 个入口的 AST 调用链必须到达共享 renderer；运行时替换共享 renderer 后实际入口必须交付替换结果；扫描 `keytao_bot/**/*.py` 的消费端，禁止已知常用度判词硬编码。允许名单逐函数限定为两个证据生产器、共享旧记录回退文案，以及四个非渲染器（解析规则、模型提示、审核诊断），没有整文件放行。新增文件也会被扫描，不依赖开发者主动补到清单。

两个原有文案断言同步更新：S63 交付用例现在要求“单边 BCC 收录不决定高低”且不含“常用度信号不足”；S57 同码提示要求真实比较器给出的 `1000 vs 10` 依据。未删用例、未跳过失败。

### 红测

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s65_commonness_render

AssertionError: '多领域 34（每百万 0.08）' not found in '常用度评估：「敲不死」与「情报所」的常用度信号不足，按空位 qbso 推荐'

Ran 1 test in 0.003s

FAILED (failures=1)
```

原始日志：`/tmp/keytao-s65/red.log`。这是修复前的事故复现；随后扩充测试使用正确的 `success` 和 `candidate_ordering_by_word` fixture 参数后完成全入口验证。

### 最终结果

六套离线测试、`e2e.test_safety`、当前全部 S62 / S63 / S64 模块，以及 S65 全部通过。自定义脚本的 `Results` 为断言计数；`Ran ... tests` 为 unittest 方法数，二者不混称。

```text
.venv/bin/python e2e/offline_checks.py

test_state_machine.py
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED

test_memory_safety.py
Ran 407 tests in 193.880s

OK

test_security_fixes.py
Results: 268/268 passed, 0 failed

test_review_gate.py
Results: 443/443 passed
✅ ALL TESTS PASSED

test_llm_policy.py
Ran 11 tests in 0.087s

OK

test_word_discovery.py
Results: 290/290 passed
✅ ALL TESTS PASSED

-m e2e.test_safety
Ran 107 tests in 0.635s

OK
```

七个独立进程全部 exit 0。原始日志和退出码索引：`/tmp/keytao-s65/final-logs/`、`six-and-safety-results.json`。suite 日志中的拒绝、error 等是负向 fixture 的预期输出，以最终断言结果为准。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s62_selection test_s62_incident test_s62_adversarial test_s62_stale_draft test_s62_reading_resolution test_s62_draft_flow test_s62_commonness

Ran 51 tests in 0.299s

OK
```

S63 importer 包含 shell 替身 fixture，需要其专用隔离入口；没有放开普通 launcher 的子进程限制：

```text
env -i PATH="$PWD/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin" LANG=en_US.UTF-8 TMPDIR=/tmp .venv/bin/python -m e2e.s63

test_s63_bcc
Ran 22 tests in 0.182s

OK

test_s63_bcc_ingest
Ran 10 tests in 0.980s

OK

test_s63_bcc_delivery
Ran 3 tests in 0.032s

OK
{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": true, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 0}, {"module": "test_s63_bcc_delivery", "exit": 0}]}
```

原始日志：`/tmp/keytao-s65/s63.log`。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s63_fresh_selection test_s63_general_reply test_s64_existing_query test_s64_explicit_submit test_s65_commonness_render

Ran 54 tests in 0.869s

OK
```

最终 S65 独立运行，以及受影响的历史候选/广告/调序入口定向回归：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s65_commonness_render

Ran 10 tests in 0.658s

OK

.venv/bin/python e2e/offline_checks.py -m unittest test_s65_commonness_render test_s54_renderer test_s57_reorder test_s58_advertisement

Ran 47 tests in 0.850s

OK
```

以上均 exit 0，原始日志及相应 `*-results.json` 已保存在 `/tmp/keytao-s65/final-logs/`。`git diff --check` 无输出、exit 0；结束 HEAD 仍为 `27663cc`，没有暂存或提交。

## 交付边界

**ZERO paid model calls：付费模型调用 0，真实 provider 调用 0，生产密钥读取 0。** 未运行 `pnpm test`、未使用 SSH、未查询生产 DB、未启动服务或部署。

常规测试通过 `e2e/offline_checks.py` 启动：清空继承凭据，禁止网络、真实凭据文件和非隔离子进程；未运行会触及凭据路径的 self-test。S63 使用仓库 `e2e.s63` 的 fixture 专用入口，以 `env -i` 空环境启动；其启动命令测试使用临时 `uv` shell 替身，只记录参数，未执行实际 builder、ingest、bot、安装或 Docker。

证据等级：源码检查 + 隔离 SQLite fixture / 假工具执行 + 指定离线套件。生产事实仅引用用户给出的事故证据；没有以本地测试冒充生产验证。改动仅在本地，未提交、未推送、未部署。
