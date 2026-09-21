# S64：已收录词查询的交付回归

已修复可逐字复现事故兜底的渲染缺陷；契约门禁和广告检测规则均未放宽。改动仅在本地，未提交、未推送、未部署、未做生产验证。

起始和结束 HEAD：`e08707298ea1799a2ff902aff7da60f60db5420b`，`fix: give the BCC ingest its full per-file timeout at startup`。起始工作树干净。

六套离线测试、`e2e.test_safety`、全部 S62、S64 回归通过。S63 执行完成，但不能宣称全绿：有一项既有测试仍期待启动超时 `15`，HEAD 已改为 `120`；四个退出码子场景失败，未修改 HEAD 副本具有完全相同的失败集合。

## 根因、提交归因和证据边界

确定的缺陷是：已有 `PendingAddWord` 时，已收录词查询保留该状态，却无条件渲染只适用于 `PendingTrustedWordRecord` 的 `回复「换码」`。

1. `keytao_bot/plugins/chat_commands.py:3678` 只在状态为空或为 `PendingTrustedWordRecord` 时保存新查询的可信词条记录；保留现有候选/操作状态本身是必要的。
2. 修改前同文件 `:3694` 只检查 `_prepared_scopes is None and len(existing_codes) == 1` 就展示换码提示，没有核对实际记录。现在对应 `:3706`，新增核对在 `:3691-3707`。
3. `keytao_bot/utils/pending_confirmation.py:855` 把它渲染为 `仍可选择其他编码（回复「换码」）`。
4. `keytao_bot/plugins/chat_routing.py:696` 的 `_pending_trusted_word_action_matches` 要求完整且非 `context_only` 的 `PendingTrustedWordRecord`。`PendingAddWord` 不能绑定裸“换码”。
5. `keytao_bot/plugins/openai_chat.py:2100-2118` 因此拒绝该命令；`:2744-2768` 又无法从这种无候选列表的简短回复选择合适的重画分支，最终发出事故中的兜底。

`bindings=0` 是 `advertised_batch_binding_pairs(text)` 提取到的正文“词→编码”对数量，不是状态记录数量。本例的解析结果为：

```text
state=PendingAddWord
command_suggestions=('换码',)
displayed_binding_pairs=()
branch=replace_missing_state state=PendingAddWord bindings=0
```

`<code>` 没有被当作可执行命令。此路径没有加入 BCC 常用度或历史频道文案，因此也不存在把这些文案误识别成命令的问题。

**使此缺陷开始触发门禁的是 `5fbc6999dbd62e97052f6c78df5537898007c043`，不是本轮 BCC 提交。** 该提交在 `keytao_bot/utils/pending_confirmation.py:2184`（该提交行号；当前 `:2202`）给 `_COMMAND_SUGGESTION_LEAD_RE` 加入 `回复|发送(?!者)`，正确捕获了以前漏检的裸“换码”广告。其父提交是 `9909dbf`。不应回退这项广告检测保护；修复应落在渲染端。

通过 `git archive` 在 `/tmp/keytao-s64/history/` 建立独立快照，使用相同假工具查询、相同初始候选状态、各快照自身的 renderer/gate 执行，结果如下。没有 checkout、reset 或修改主工作树历史。

| 完整代码快照 | 无旧候选状态 | 已有 PendingAddWord |
| --- | --- | --- |
| `52e0f346` | 保留已收录回复 | 保留回复，尚未检测出该广告 |
| `e983acf` / `b1b33ce` / `86f9ea7` | 保留已收录回复 | 保留回复，尚未检测出该广告 |
| `9909dbf`（`5fbc699^`） | 保留已收录回复 | 保留回复，尚未检测出该广告 |
| `5fbc699` | 保留已收录回复 | 事故中的原样兜底 |
| `bb01a7e` | 保留已收录回复 | 事故中的原样兜底 |
| `67ddb7a` | 保留已收录回复 | 事故中的原样兜底 |
| `faed1da` | 保留已收录回复 | 事故中的原样兜底 |
| `e087072` | 保留已收录回复 | 事故中的原样兜底 |

历史探针：`/tmp/s64_history_probe.py`；每个快照的原始输出为 `/tmp/keytao-s64/history/<hash>.log`。探针输出原始回复和 gate 后回复，其 unittest `OK` 只表示探针执行成功，不代表旧行为正确。

生产日志未包含 gate 前完整正文、候选状态内容或它的创建回合，故不能唯一还原生产回合的所有分支，也不能断言该记录由这轮 BCC 新建。这里确认的是与日志中的状态、分支、bindings 数和最终回复一致的缺陷，并用完整旧版本对照确定其引入点。没有以本地复现冒充生产回放。

## 影响范围

不是每个已收录词查询都会失败，也不限于二字词或 BCC 命中词。

- 确认受影响：已收录、只有一个编码、回复广告裸“换码”，但当前保留了无法绑定该命令的候选状态。候选属于同一个词或另一个词均可触发。
- 不受这个缺陷影响：无旧状态且成功保存可信词条的简短已收录查询；完整服务端候选列表配套 `PendingAddWord` 的渲染也能通过门禁。
- BCC 未安装、已安装但未命中、现代频道命中、仅历史频道命中时，上述区别相同。
- 单字、二字词和三字词都在回归矩阵中覆盖。`乔布斯` 在 `bb01a7e` 曾成功，与本结果不矛盾：是否持有兼容状态才是触发差异。

## 修复

- `keytao_bot/plugins/chat_commands.py:3691`：读取实际记录，要求没有正在执行的操作、词和编码匹配，并由真实 `_pending_trusted_word_action_matches(..., "换码")` 证明可绑定后，才显示“回复「换码」”。不删除、不替换其他待处理候选或票据。
- `keytao_bot/utils/pending_confirmation.py:863`：未展示上下文控制时，追加编码格式明确包含词名，避免用户填入编码后绑定到旧状态里的另一个词。它仍是需要填写实际编码的格式说明，不是虚构的可执行计划。
- `test_s64_existing_query.py:87`：36 个子场景，实际查询渲染 → 回复规范化 → 编码补充 → 交付门禁 → 最终用户回复。覆盖四种 BCC 条件 × 三种词长度/类型 × 三种旧状态条件，核对状态保留、事实、操作格式及没有写工具调用。
- `test_s64_existing_query.py:146`：负向用例继续将原来不绑定的“换码”广告拒绝为完全相同的兜底；门禁未变松。
- `test_s64_existing_query.py:157`：真实 `AgentOrchestrator` 和 `ToolExecutor` 接收六种事故工具结果，三次预录假模型响应创建 `PendingAddWord`，核验完整候选回复；再查询同词，检查简短已收录回复也能通过门禁。四种 BCC 条件均执行。

修复后的旧候选状态场景，回复主体为：

```text
「单份」已在词库（dffn）。
仍可选择其他编码；若保留现状，无需操作。
可再追加一个编码，格式为：给 单份 加一个码 <code>（将 <code> 换成实际编码）。
```

无旧候选且可信记录完整时，原来的“回复「换码」”和 `加入编码 <code>` 均保留。完整候选列表仍保留编号/编码选择、加入和加入并提交。

BCC 数据来自仓库 S63 slice 的临时 SQLite 副本；针对“单份”的现代/历史命中计数明确是合成测试数据，不是声称生产词频如此。单字扩展用例也是类型覆盖夹具。测试没有加载生产语料库或密钥。

## 执行结果与原始 tails

付费模型调用 0，真实 provider 调用 0，生产密钥读取 0。未运行 `pnpm test`。常规测试使用 `e2e/offline_checks.py` 的环境白名单、网络/真实凭据文件/非隔离子进程禁用机制；未执行其凭据路径 self-test。

### S64 红 → 绿

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s64_existing_query
```

最初先在修复前运行最小复现：`Ran 1 test in 0.048s` / `FAILED (failures=1)`，记录为 `/tmp/keytao-s64/red-existing.log`。

最终同一回归文件复制到未修改的 `e087072` 快照后，完整红测如下（24 个矩阵子场景 + 4 个真实 orchestrator 建状态后再查询的子场景）：

```text
Ran 3 tests in 0.149s

FAILED (failures=28)
```

当前修复版本，最终绿测：

```text
Ran 3 tests in 0.149s

OK
```

完整日志：`/tmp/keytao-s64/red-final-tests.log`、`/tmp/keytao-s64/final-logs/m-unittest-test_s64_existing_query.log`。这里的 3 是 unittest 方法数；子场景数量另计。

### 六套离线测试和 safety

```text
.venv/bin/python e2e/offline_checks.py
```

全部 exit 0；以下为各日志结尾的原始结果行。`Results` 是自定义 suite 的断言计数，不混称 unittest 方法数。

```text
test_state_machine.py
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED

test_memory_safety.py
Ran 407 tests in 193.896s

OK

test_security_fixes.py
Results: 268/268 passed, 0 failed

test_review_gate.py
Results: 443/443 passed
✅ ALL TESTS PASSED

test_llm_policy.py
Ran 11 tests in 0.098s

OK

test_word_discovery.py
Results: 290/290 passed
✅ ALL TESTS PASSED

-m e2e.test_safety
Ran 107 tests in 0.614s

OK
```

原始日志和退出码索引：`/tmp/keytao-s64/final-logs/`、该目录的 `six-and-safety-results.json`。

### S62

覆盖当前全部七个 `test_s62_*.py` 模块，exit 0：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s62_selection test_s62_incident test_s62_adversarial test_s62_stale_draft test_s62_reading_resolution test_s62_draft_flow test_s62_commonness

Ran 51 tests in 0.300s

OK
```

### S63

起初将 S62/S63 合在同一条 launcher 命令，因日志文件名超过文件系统限制而报 `OSError: [Errno 63] File name too long`，测试尚未开始；随后拆分。

S63 五模块首次使用常规 launcher 执行，四个 shell 子场景被更严格的子进程保护挡住：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s63_bcc test_s63_bcc_ingest test_s63_bcc_delivery test_s63_fresh_selection test_s63_general_reply

Ran 60 tests in 0.304s

FAILED (errors=4)
PermissionError: S60 offline guard: unguarded child disabled
```

接着使用已有的 S63 fixture 专用入口，外层以空环境启动。该入口禁止网络和生产凭据路径；启动测试的 `uv` 是临时 shell 替身，只记录参数，不执行 builder、ingest、bot、Docker 或安装命令。

```text
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR=/tmp LANG=en_US.UTF-8 .venv/bin/python -m e2e.s63

test_s63_bcc
Ran 22 tests in 0.180s

OK

test_s63_bcc_ingest
Ran 10 tests in 1.352s

FAILED (failures=4)

test_s63_bcc_delivery
Ran 3 tests in 0.032s

OK
{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": false, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 1}, {"module": "test_s63_bcc_delivery", "exit": 0}]}
```

既有失败原因：`Dockerfile:15` 在 `e087072` 改成 `--timeout 120`，但 `test_s63_bcc_ingest.py:125` 仍断言 `--timeout 15`。同一 fixture 入口在未修改的 `e087072` 副本上也得到：

```text
Ran 22 tests in 0.180s

OK
Ran 10 tests in 1.145s

FAILED (failures=4)
Ran 3 tests in 0.033s

OK
```

逐个比对失败标识，相同集合为 `test_s63_bcc_ingest.BccIngestTests.test_container_command_starts_bot_after_any_ingest_exit` 的 `exit_code=0, 1, 2, 137`；新增失败 `[]`，消失失败 `[]`。没有顺带修改启动配置或该既有测试。

独立执行另外两个 S63 模块，exit 0：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s63_fresh_selection test_s63_general_reply

Ran 25 tests in 0.156s

OK
```

原始日志：`/tmp/keytao-s64/s63-fixture.log`、`/tmp/keytao-s64/baseline-s63-fixture.log`，以及 `/tmp/keytao-s64/final-logs/` 内各模块组日志。

### 相邻格式和安全回归

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s57_extra_code test_s58_advertisement test_s54_renderer test_s64_explicit_submit

Ran 61 tests in 0.469s

OK

git diff --check
exit 0
```

## 处置建议与交付边界

建议采用本次前向修复，不建议回退 `67ddb7a` 或 `faed1da`：`bb01a7e` 已能复现同样的问题，回退 BCC 不能消除根因。回退 `5fbc699` 会失去扩大后的广告检测保护、GLM provider 策略、移码语法和 grammar-gap 票据桥接，范围明显大于这个渲染修复。

本次只修改两个产品文件，新增回归文件和本报告；没有修改门禁、授权语法、BCC 规则、工具 schema、依赖或部署配置。结论止于静态证据和离线执行。实际部署及生产确认由用户执行。
