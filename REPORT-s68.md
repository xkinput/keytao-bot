# S68：限制副词选择与审词提案桥接

本地修复及验收已完成。六套离线测试、`e2e.test_safety`、全部指定历史回归和 S68 fixture 均通过。所有修改只在工作区，没有提交、推送或部署。

起始 `git log -1 --oneline`，与要求一致；起始工作区干净：

```text
b73623f fix: name every figure's source, and never advertise a reorder that changes nothing
```

## 缺陷结论与证据

| 项目 | 结论 | 实现与验证 |
| --- | --- | --- |
| A：限制副词与单词裸编号 | FIXED | `pending_confirmation.py:1047` 接受动作前的 `只 / 仅 / 就 / 光 / 只要`；`authorization_grammar.py:2995` 支持带词选择的相同修饰；`chat_routing.py:1920` 将单词 batch 记录的裸选择绑定到唯一词，整个查询涉及多个词时仍要求词名。 |
| A：跨读音编号 | 已有全局编号映射保留，新增回放锁定 | `test_s68_selection.py:101` 经真实阶段链、dispatcher、ToolExecutor，第一读音 1–3、第二读音 4–6，3 精确写入 `乐句@yhjluu`；T2 不调用模型。 |
| A：与 subset-submit 冲突 | REBUTTED | `_SUBSET_SUBMIT_RE`（`chat_commands.py:7520`）严格要求 `只 / 仅` 后接 `提交 / 提审`，没有匹配“只加入”的路径；S68 专门测试这一边界，没有改动 S61 规则。 |
| B：精确选择与排除说明 | FIXED | `chat_commands.py:10734` 在生产阶段与核心入口共用的执行函数中附加一行未选择的编号、词、码；两词场景沿用 S62 的未选择词说明。 |
| C：batch-add 完全未实现桥接 | 部分 REBUTTED，实际遗漏 FIXED | 原有 `tools.py` 已有 batch-add 协议与 `test_s58_generic_bridge.py:108` 测试。实际遗漏是意图探测必须出现字母编码、Create 还要求词和码出现在原文、batch 参数拒绝 `remark`。本次修复这些入口条件。 |
| D：错误否认请求 | FIXED | `orchestrator.py:2593` 不再返回“当前消息没有明确要求执行”；`tools.py:1479` 对桥接失败说明核对失败，并提供一条可执行的重新审词命令。 |

以上路径均相对仓库；`pending_confirmation.py` 位于 `keytao_bot/utils/`，`authorization_grammar.py`、`tools.py`、`orchestrator.py` 位于 `keytao_bot/harness/`，`chat_routing.py`、`chat_commands.py` 位于 `keytao_bot/plugins/`。

## 新增语法与“只”的含义

只在完整候选选择语法中识别限制副词，不做任意子串替换。动作可以在前或在后；保持半／全角逗号、顿号、分号，忽略结尾句点，裸编号还做 NFKC 归一化。

真实阶段链回放的输入包括：

```text
只加入3
仅加入 3
只添加3
加入3
3
就加入3
光加入3
只要加入3
3，只加入。
只加入，３．
只加入3、4
仅添加３；４。
3,4,只加入
只要加入 3;4.
```

上述单选写法均绑定当前记录的第三个候选，不依赖推荐项或模型重新选择。多选严格保留列出的集合；未列出的推荐项不会补进来，也不会隐式提交。两词列表的 `只加入 乐句 3`、`仅添加 乐句 3`、`乐句 3，只加入。` 仅写入乐句，提示小端未添加；两词列表的裸 `3` / `只加入3` 被拒绝，不猜测词名。

同样保留重复编号、越界、否定、转述、问句、附加删除／提交指令、原始引号、缺失服务器候选、过期记录、其他发送者以及重复消费的拒绝边界。`只提交3` / `仅提审3` 不属于候选添加语法。

## 桥接安全论证

入口在 `tools.py:1570`，仅处理 `verb_not_matched` 且当前消息有正向操作意图的情况。`authorization_grammar.py:3041` 增加无字母编码的添加意图识别，同时排除解释、转述、记录语境、否定、问句等。无字母编码的桥接只开放给 `keytao_create_phrase` 和 `keytao_batch_add_to_draft`，不扩大删除／移位／撤回的原文绑定范围。

`tools.py:1797` 的规则：

1. 只读结构化参数；词和码必须是具体字面值，限制形状、条数、唯一性、字段集合。未知字段、模型提供的 confirmed/CAS/digest/batch/internal review 字段拒绝。
2. 每一个 Create 必须精确命中本轮 `trusted_reviewed_items_by_key[(word, code)]`；核对类型、非空读音、完整候选码链、布尔审核标记。没有本轮审词、只有模型正文或历史文本、错误词／码或不完整 capability 都不能出票。
3. batch 的字符串 `remark` 可以随模型参数出现，但内容完全丢弃；提交提案中的备注、读音、码链、审核标记从可信 capability 重建（`:1843`、`:1859`）。单条 create 的原有严格参数约束保留。
4. 重新查询当前词库（`:1875`），要求返回完整、逐词对应、类型正确的记录，拒绝重复或歧义。Change/Delete 继续要求原文操作数与现有身份匹配。
5. 只调用 `preview_only=True, confirmed=False`；拒绝失败项、跳过项、重新规划、缺封印、意外 success 等不完整结果。只有完整 server-warning ticket 能保存，输出 `grammar_gap_bridged=True`（`:1788`），不会自动确认。
6. orchestrator 在展示确认文案前保存真实票据，并结束模型循环（`orchestrator.py:2541`）。后续确认仍使用既有发送者、TTL、一次性消费、版本和摘要校验。

八个 mutation 工具的范围核对：

| 工具 | 最终边界 |
| --- | --- |
| `keytao_batch_add_to_draft` | 新增无原文词码的已审 Create 提案；Change/Delete 保留 S58 的字面操作数检查。 |
| `keytao_create_phrase` | 同样支持已审 Create capability；与 batch 共用验证。 |
| `keytao_shift_phrase_code` | 保留原 S58 的唯一现有词条、服务器 shiftPlan 与摘要检查。 |
| `keytao_remove_draft_item` / `keytao_batch_remove_draft_items` | 保留原 S58 的字面 ID、当前发送者草稿快照、完整 deleteTargets 预览协议。 |
| `keytao_update_draft_item_weight` / `keytao_recall_batch` | 没有本桥接所需的完整只预览确认协议，继续拒绝；不能为了出票调用会直接修改状态的工具。 |
| `keytao_submit_batch` | 候选审词记录不授权整批提交，保留独立的批次提交预览／确认路径，不从添加语法缺口获得提交提案。 |

`test_s68_selection.py:265` 用真实 orchestrator 和 fake model/fake tools 回放“本轮 prepare_reviewed_add → literal batch-add”。它验证保存票据、只预览、类型／读音封印、正文伪造不进入票据、外部发送者无票据；去掉本轮审词或改为不可核对编码时，没有 mutation sink 调用或票据。`:235` 覆盖注入字段、附件、保护语句、非字面操作数和不完整预览；已有 S58 再覆盖确认的精确重放。

## T2 前后

用户提供的生产现象：`只加入3` 落入 `flow=general`，4 次模型调用后 batch-add 被 `verb_not_matched` 拦截，回复“当前消息没有明确要求执行这项操作”。本任务没有访问生产核验该日志。

本地 fixture 的 T1 保留双读音 1–6 编号和用户提供的 BCC 比值 `11.25` / `6.28`。第二读音码链使用合成 fixture，未将它当作新的生产词库事实。T1 的查词意图是 fixture，T2 的选择、路由、dispatcher 和 executor 使用真实代码；全部网络边界为 fake tool。

修复后 `/tmp/keytao-s68/replay.json`：T2 `flow=pending-confirmation`、`modelCalls=0`，唯一写入为 `乐句@yhjluu`，未提交；交付文案为：

```text
✅ 已确认添加到草稿
草稿内容：乐句 yhjluu
草稿地址：https://keytao.rea.ink/batch/s68-fixture
未选择：1. 乐句 → yhjl、2. 乐句 → yhjlu、4. 乐句 → lejl、5. 乐句 → lejlu、6. 乐句 → lejluu；这些候选本次未添加。
```

该链接是 fixture 批次标识，不是真实生产批次。

最初红测保存于 `/tmp/keytao-s68/red.log`。另外用 `git show b73623f:...` 提取原始 `_strip_selection_action` 和 `looks_like_mutation_grammar_gap`，在最终 fixture 内分别替换这两个函数，避免早期 T1 stub 不完整影响定位；这是函数级差分复现，不是整个旧 checkout 的验收：

```text
.venv/bin/python e2e/offline_checks.py -m unittest discover -s /tmp/keytao-s68 -p test_baseline.py

Ran 2 tests in 0.123s

FAILED (failures=2)
```

第一条在 T2 进入 general model 路径时触发禁止模型调用断言；第二条直接返回 `blockReason=verb_not_matched`，没有 `grammar_gap_bridged`。没有实际模型请求。

## 回归期间发现及处理

- 状态机的 4 个旧断言固定为 `added` / `shifted` 等短字符串，现更新为包含完整排除行的精确字符串；原有词码、调用次数、提交标记断言保留。
- 内存安全首轮 `407` 方法中出现 `8` 个失败子项：4 项是旧回执／拒绝文案；另外是扩大到删除路径导致转述语料出现不应有的读取尝试，以及 3 个旧安全建议的入口变化。收窄到已审添加并接入记录语境守卫后，原有安全断言不改，28 个相关定向方法通过。首轮完整日志：`/tmp/keytao-s68/first-memory-safety.log`。
- 更新 S63 和内存安全对旧“没有明确要求执行”的断言，改为核对失败原因、单一拒绝、可执行命令，并继续断言零写入。
- 额外对抗负例发现 `会议记录：只加入3` 内部指令不能完整解析时，旧的记录语境判断还不足以拦住提案。最终改为直接复用记录标记及“是否为词条操作数”的判断，不依赖内部指令解析成功；`会议记录`、`记下`、`原话` 三个负例均通过。
- S63 BCC ingest 的 shell fixture 在通用离线 launcher 下被禁止未隔离子进程的守卫阻止。改用仓库已有 `env -i ... python -m e2e.s63` 专用禁网入口；临时 `uv` 是 fixture shell 替身，只记录命令并返回给定退出码，不执行 builder、ingest、bot 或 Docker。原失败保留在 `/tmp/keytao-s68/first-regressions/group-7.log`，没有放宽通用守卫。

## 验收尾部

### 六套离线测试与 safety

以下最终一次完整运行 exit 0，七个独立子进程全部 exit 0：

```text
.venv/bin/python e2e/offline_checks.py
```

`.venv/bin/python test_state_machine.py`：

```text
============================================================
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED
============================================================
```

`.venv/bin/python test_memory_safety.py`：

```text
Ran 407 tests in 217.786s

OK
```

`.venv/bin/python test_security_fixes.py`：

```text

============================================================
Results: 268/268 passed, 0 failed
============================================================
```

`.venv/bin/python test_review_gate.py`：

```text

============================================================
Results: 443/443 passed
✅ ALL TESTS PASSED
```

`.venv/bin/python test_llm_policy.py`：

```text
Ran 11 tests in 0.113s

OK
```

`.venv/bin/python test_word_discovery.py`：

```text

============================================================
Results: 290/290 passed
✅ ALL TESTS PASSED
```

`.venv/bin/python -m e2e.test_safety`：

```text
Ran 107 tests in 0.743s

OK
```

日志与退出码归档：`/tmp/keytao-s68/final-logs/six-and-safety-results.json`。源码指纹复核一致；最终 `git diff --check` 通过，HEAD 仍为 `b73623f`，分支 `main`，没有暂存或提交。

### 指定历史回归

全部 exit 0；S63 的三个 Ran 尾部分别来自 bcc、bcc_ingest、bcc_delivery。

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s54_multiword test_s54_renderer test_s54_selection

Ran 37 tests in 0.116s

OK
```

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s58_bridge test_s58_generic_bridge test_s58_grammar test_s58_advertisement test_s58_explicit_destination

Ran 62 tests in 0.413s

OK
```

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s59_commonness_evidence test_s59_commonness_route

Ran 18 tests in 0.079s

OK
```

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s59_tool_prompt

Ran 6 tests in 0.078s

OK
```

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s59_general_cap

Ran 11 tests in 0.120s

OK
```

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s62_selection test_s62_incident test_s62_adversarial test_s62_stale_draft test_s62_reading_resolution test_s62_draft_flow test_s62_commonness

Ran 51 tests in 0.331s

OK
```

```text
env -i PATH=/Users/rea/code/keytao-org/keytao-bot/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin LANG=en_US.UTF-8 TMPDIR=/tmp /Users/rea/code/keytao-org/keytao-bot/.venv/bin/python -m e2e.s63

Ran 22 tests in 0.199s

OK

Ran 10 tests in 1.020s

OK

Ran 3 tests in 0.037s

OK
```

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s63_fresh_selection test_s63_general_reply

Ran 25 tests in 0.165s

OK
```

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s64_existing_query test_s64_explicit_submit test_s65_commonness_render test_s66_readings test_s67_ranking test_s68_selection

Ran 70 tests in 1.678s

OK
```

S63 最终摘要：

```text
{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": true, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 0}, {"module": "test_s63_bcc_delivery", "exit": 0}]}
```

### S68 单独回归

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s68_selection

Ran 13 tests in 0.392s

OK
```

最终回归日志归档：`/tmp/keytao-s68/final-logs/`；命令与退出码见 `regressions.json`。

## 预算与交付边界

**ZERO paid model calls：付费模型调用 0，真实模型/provider 调用 0，生产密钥读取 0。**

未运行 `pnpm test`、在线 E2E、真实工具写入、部署、commit 或 push。常规测试使用 `e2e/offline_checks.py` 的清空凭据环境、禁网和凭据文件审计守卫；S63 使用其既有 fixture-only 入口。未执行会尝试打开真实凭据路径的 guard self-test。离线通过不代表生产验证。
