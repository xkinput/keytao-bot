# S62：裸选择写入草稿、选择与动作合并、草稿展示及读音消歧

日期：2026-09-11。范围：本地实现、fixture/fake-tool 验证；未提交、未部署、未访问生产。

## 结果与基线

- FIXED A/B：选择支持首尾动作；逐项绑定，保留可执行词条并解释跳过原因；裸选择直接写入草稿，返回逐词变更清单和批次链接，不提交。
- FIXED C：`查看草稿` 保留真实草稿内容，空草稿显示零条；不会附上旧候选确认提示再被替换。
- FIXED D：明确编码或读音匹配可信候选后，保留采用读音和人工审核标记；同码读音不再无条件阻塞。
- FIXED E：保留重复抑制，改为原因和一个可执行命令；本轮选择说明也经过最终交付校验。
- REBUTTED 基线：实际初始工作树干净，HEAD 为 `0176089`，而非简报的 `1814b6c`。已有 `5eef57c`、`3c9b81d`、`0176089` 三个后续提交全部保留，本轮没有 Git 写操作。
- REBUTTED “无动作行为不变”：前一轮按简报改为 selection-only，属于行为变化；该事实异议已被接受。本次以 `0176089` 重新核对，恢复裸选择写入草稿的产品语义，并同步全部相关断言。

## Follow-up：0176089 的实际行为与本次修正

本次开始时 HEAD 仍为 `0176089fde9fb34a05a34b6da49e6f021731c5c1`、分支 `main`，工作树已有 S62 未提交改动。开始快照和本次增量证据保存在 `/tmp/keytao-s62-followup/`，没有恢复、覆盖或暂存无关改动。

`git show 0176089:keytao_bot/plugins/chat_routing.py` 的裸选择成功路径原文：

```python
# chat_routing.py:1971-1978 at 0176089
return (
    PendingToolConfirm(
        function_name=state.function_name,
        args=derived_args,
        confirmation_source=state.confirmation_source,
    ),
    MessageCommandIntent(intent="pending_confirm", confidence=1.0),
    None,
)
```

这个 `pending_confirm` 是执行意图，不是“只保存选择，再等用户加入”。同版本 `chat_commands.py:12396` 调用 `_execute_confirmed_tool`；`:6875` 的 `batch_warning_confirmation_binding(data, args)` 在服务端预览绑定完整且可自动确认时，于 `:6885` 递归执行确认；成功在 `:7088` 后构造：

```python
# chat_commands.py:7092-7095 at 0176089
partial = failed_count > 0 or success_count != expected_count
header = "⚠️ 仅部分加入草稿\n" if partial else "✅ 已加入草稿\n"
response = header + await _format_draft_response(
    data, platform, user_id, compact=True,
)
```

因此，`0176089` 中可绑定的裸选择会尝试写入所选词；完整、无新增风险的预览会在同一回合完成写入并给正常回执，未知警告仍遵循既有票据规则。它不因缺少“加入”而停止，也不会凭裸选择提交。基线 `test_s62_selection.py:85-91` 明确遍历 `加亮 jslxa`、`加亮 3`、`加亮 添加 3`，断言只写 `加亮/jslxa`，工具调用两次。

本次删除 `_selection_only` 的生成、授权否定、保存后提前返回和专属确认文案/交付分支。当前 `chat_routing.py:1745` 将有效选择视为授权，`:2035` 在无提交动作时给出 `pending_confirm`；`chat_commands.py:12207` 采用所选集合后继续既有执行器。提交仍要求明确提交指令；batch version、warning digest、快照、owner、一次性执行及审核封签沿原路径。

直接固定本次语义的测试：

- `test_s62_selection.py:85`：恢复两种裸输入，核对单次确认写入、清空完成票据、逐词清单、真实 fixture 批次链接、没有提交回执。`:98` 覆盖两词裸编码/编号；`:123` 覆盖裸选择的缺失/篡改 seal 和 server-warning 阶段换选拒绝。
- `test_s62_incident.py:207`：从真实 turn-2 discovery，经真实 dispatcher/ToolExecutor 到最终交付，分别运行 `敲不死 2` 和 `鸡白汤 jbtaua, 敲不死 qbso`；确认 create preview/confirm 各一次、submit 调用零次、人工审核标志保留、逐词回执及链接、鸡白汤因读音缺失未加入的说明。`:226` 固定非推荐的 `敲不死 3` 当回合写入。
- `test_s54_selection.py` 的授权断言、`test_s54_renderer.py` 的候选提示、`test_s54_multiword.py` 的 SQLite 重载/actor 隔离、`test_s63_fresh_selection.py:372` 的 live pair，均已同步直接写入；原有其它语义未回滚。
- `test_s62_adversarial.py:37` 和 `test_s62_stale_draft.py:141` 固定裸选择最终回执；错误说明 provenance 的正文、消息、快照、当前状态篡改与失效检查继续保留在 `test_s62_stale_draft.py:162`，改用无效编号作为来源。

### 占用码的例外按词处理

FIXED：复核发现，先前 reviewed 多词选择在遇占用码时只设人工审核、随后清空调序建议；本仓库 batch sink 在 `keytao_bot/skills/keytao-draft/tools.py:4871` 校验编码后发 batch 请求，不能当作已执行常用度保护的证据。本次在 `chat_routing.py:1991` 复用 `candidate_inventory.py:20` 的 `protected_candidate_occupants`，输入为通过镜像一致性校验的原始候选、占位词和常用度判断，检查发生在派生集合清空调序建议之前。

只有与词、编码、每个占位词精确匹配的 `front_more_common` 才放行占用码，并继续携带人工审核标记；`behind_more_common`、`close`、无足够证据、错词/错码判断或占用信息不完整，都对该词解释并跳过。没有宣称证据不足等同于占位词肯定更常用，也没有让一个受保护词挡住其它有效选择。

`test_s62_selection.py:142` 固定上述判断及允许分支，`:172` 固定混合顺序下只写另一个合法词、保留原因及批次链接。`test_s54_multiword.py` 的 SQLite 测试先验证占用码受保护，再选择空位写入。新 commonness 回归修复前为 `Ran 2 tests ... FAILED (failures=8)`，日志 `/tmp/keytao-s62-followup/red-commonness.log`；修复后通过。裸选码最初的失败回归为两项空写入失败；同次测试还有三项错误的 fixture 手工审核字段断言，已移到真实 dispatcher 的写入用例核验，不把那三项算作产品回归。完整初始记录为 `/tmp/keytao-s62-followup/red-selection.log`。

## 语法与绑定

入口：`keytao_bot/harness/authorization_grammar.py:2986`，`parse_reviewed_selection_command`。

```text
selection := pair (separator pair)*
command   := selection | action separator selection | selection separator action
pair      := word whitespace [reading ':'] (number | code)
separator := whitespace-between-action-and-selection | ',' | '，' | '、' | ';' | '；'
action    := 加入 | 加入草稿 | 加到草稿 | 写入草稿
           | 加入并提交 | 加入草稿并提交 | 加到草稿并提交
```

中文冒号同样可用；结尾 `。` / `.` 忽略。词条之间仍须使用列出的标点；保留已有的 `词名 添加 编号/编码` 形式。
纯选择授予所选词的草稿写入权限；同步推荐编码及审读镜像后，直接执行所选集合。未选择词保持未添加。非推荐编号也在本回合写入。

原整条拒绝的证据（`git show 1814b6c:...`）：

- `keytao_bot/plugins/chat_routing.py:1928`：纯 pair parser 无法接受动作尾句，直接返回旧“选择格式”文案。
- 同文件 `:1936`、`:1940`、`:1945`：未知词、越界编号或不在候选内的编码立即 return，导致前后有效选择全部丢弃。

修复在当前 `chat_routing.py:1904`：完整解析后，逐项构造准确的派生集合，跳过项写入 `_selection_skipped`；动作包含提交时进入既有 batch add/submit 协调器，保留预览、版本/摘要绑定和确认重放。
例如最终交付已实测：

```text
已按你的选择加入：敲不死 → qbso；「鸡白汤」还没确定读音，未加入。
已变更：敲不死 → qbso
✅ 批次已提交审核。
草稿地址：https://keytao.rea.ink/batch/s62-fixture
```

该地址为测试夹具字符串，未访问对应站点。混合消息的鸡白汤没有 live 可写记录时按 B 跳过；另一个 fresh 显式编码请求会重新核验候选并按 D 写入。

授权边界：重复词、附加操作、问句、嵌套引号、损坏的候选镜像和风险确认阶段换选仍拒绝。对未绑定词携带的否定/转述前缀先拒绝整条封套，不能跳过该前缀后误写其余词；真实 live 词名“不明觉厉、记忆力、说唱”仍可选择。独立检查的错误写入反例已有零工具调用回归（`test_s62_adversarial.py:16`）。

## 草稿替换与本轮说明交付

`1814b6c` 的 `openai_chat.py:5775` 在 `_stage_append_ticket_challenge` 中只排除 recall/clear，没有排除 draft_view。实际草稿回复被附加旧 pending 的操作提示；随后 advertised reply contract 不接受该混合回复，经 `branch=replace_from_live_state`（旧 `:2709`）重画整份旧候选。

当前 `openai_chat.py:5887` 将 draft_view 排除出旧票据提示追加。`test_s62_stale_draft.py:27` 从真实草稿路由、两次 fake 查询一直运行到最终 bot.send；空/有草稿均保持查询正文，pending 候选原样保留。

独立检查曾发现选择确认、越界编号说明会被同一 guard 重画。本次裸选择改走正常写入回执；`ReviewedSelectionReply` 仅保留确定性错误说明，`openai_chat.py:2524` 用原始消息与快照重跑真实 resolver，要求当前状态、完整正文严格相同且下一步命令绑定成功。普通模型文本没有该通道。正文、消息、快照、当前状态的篡改/失效均有拒绝测试（`test_s62_stale_draft.py:162`）。

## 读音规则

- `explicit_code.py:128`：按可信 pronunciation group 独立验证明确编码和可选读音，避免同一码的多组读音在扁平化时丢失。
- `explicit_code.py:158`：继续检查小写字母、最长六位、词条类型及候选音码前缀；无法验证的形码后缀标为人工复核，不虚构形码正确性。
- `chat_commands.py:11916`：只解除已明确标识为读音歧义且占用查询成功的阻塞；其他 BLOCK、缺失候选、查询失败仍拒绝。写入备注说明采用读音，并设置人工审核。
- `keytao_review.py:4088`：多组完整候选编码序列完全相同且占用查询成功时，保留第一组写入读音，全部等价读音另存 `equivalentPronunciations`；强制人工复核。候选编码不同且用户未选定时仍询问。
- 真实 prepare 对照测试在 `test_s62_reading_resolution.py:66` / `:78`；明确编码和 `jī bái tāng：jbtaua` 的真实路由、fake create 预览/确认、fake submit 预览/确认在 `:221`，均只写一次、提交一次且保留审核标记。

## 文案与闭包

重复判断机制仍保留在 `orchestrator.py:3891`；当前 `:3923` 回答已收到请求、保留失败原因，并由 renderer 输出、真实 draft parser 验证 `查看草稿`。不再输出旧“相同拒绝路径”或“已停止重复建议”状态句。

`test_s62_incident.py:254` 使用独立广告提取器读取 footer，固定核对全部四条：`加入`、`加入并提交`、`敲不死 2`、`敲不死 2，加入并提交`，逐条通过真实 parser/binding；未使用被测 parser 过滤广告。七种 action × 六种分隔符 × 首尾两个位置，共 84 个组合也检查真实绑定并执行 fake 写入/提交。

S62 重放方式：每个 turn 3/4/5/6 变体均从真实 turn-2 discovery 生成的同一候选条件开始，执行选择与提交后再执行 turn 7；首个修复后的提交会消费票据，因此没有把原事故后续失败强行套到已消费的状态上。另有实际 live pending 未消费时的草稿查询回归。`test_s62_incident.py:175` 使用真实 dispatcher/ToolExecutor，仅替换最后的 transport，验证混合选择的审读 capability 和最终交付。

## 离线验证

所有测试通过现有 `e2e/offline_checks.py` 启动：子进程使用环境白名单，禁止网络、真实凭据文件读取和未隔离子进程；只执行指定 suite/unittest fake tests。没有运行 launcher 的凭据路径 self-test。

最终六套及 safety 命令：

```text
.venv/bin/python e2e/offline_checks.py
```

该命令逐一执行以下七项，全部 exit 0；实际 suite tails 如下（自定义脚本的 Results 是脚本断言计数，不混称 unittest 测试用例数）：

```text
test_state_machine.py
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED

test_memory_safety.py
Ran 407 tests in 207.833s
OK

test_security_fixes.py
Results: 268/268 passed, 0 failed

test_review_gate.py
Results: 443/443 passed
✅ ALL TESTS PASSED

test_llm_policy.py
Ran 11 tests in 0.099s
OK

test_word_discovery.py
Results: 290/290 passed
✅ ALL TESTS PASSED

-m e2e.test_safety
Ran 107 tests in 0.644s
OK
```

以上均为本次 follow-up 在最终产品代码上的执行结果。完整日志和 launcher 退出码在 `/tmp/keytao-s60-offline/`，本次独立归档为 `/tmp/keytao-s62-followup/final-logs/`，总索引为该目录的 `all-results.json`。没有把旧日志作为本次通过证据。

S62 与相邻回归的实际命令及 suite tails：

```text
.venv/bin/python e2e/offline_checks.py -m unittest test_s62_selection test_s62_incident test_s62_adversarial test_s62_stale_draft test_s62_reading_resolution test_s62_draft_flow test_s62_commonness test_s54_selection test_s54_renderer test_s54_multiword test_s63_fresh_selection
Ran 104 tests in 0.409s
OK

.venv/bin/python e2e/offline_checks.py -m unittest test_s56_explicit_code test_s56_security test_s56_commonness test_s56_commonness_binding test_s57_extra_code test_s57_security_review test_s61_submit_replay test_s64_explicit_submit
Ran 73 tests in 0.361s
OK
```

第一条命令包含当前全部七个 `test_s62_*.py` 模块。新增常用度测试及最终交付检查均在最后这次执行中；104/73 是 unittest 用例数，不包含各用例内部的 subTest 数量。

独立只读增量审查没有剩余 blocker；审查指出常用度枚举命名可更准确，已改用实际 `behind_more_common` 并加入 `close`，随后重新运行上述 104 tests。审查是静态证据，测试执行由主会话完成。`git diff --check` 通过。

中途一次候选文案把“提交”放入命令引号，导致广告提取多出未绑定指令，已调整措辞并通过完整 footer 闭包检查。另把 incident 的真实 dispatcher 引用移到 mock 生效前导入，消除多次 replay 时捕获前一 fixture mock 的污染；真实 dispatcher 用例现在确实经过 ToolExecutor，仅最后 transport 为 fake。两项中途失败均未以放宽写入/提交计数断言处理。

### 前一轮基线旧断言的处理（本次保留）

前一轮报告记录的首轮 memory suite 为 407 tests，1 failure + 2 errors。前一轮通过仅导出 HEAD 的 `keytao_bot`、`e2e`、`test_memory_safety.py` 至 `/tmp/keytao-s62-baseline-0176089.5HbDgY`，用同一 guard 复现完全相同 nodeid 集合；以下是保留的历史证据，本次没有重复运行基线：

```text
ReadOnlyTurnToolExposureTests.test_one_reason_is_explained_once_per_turn
ReadOnlyTurnToolExposureTests.test_withheld_write_tool_answers_with_a_reason_not_a_crash
OrchestratorTrustBoundaryTests.test_memory_and_quote_cannot_authorize_model_requested_write
Ran 21 tests in 0.124s
FAILED (failures=1, errors=2)
```

基线日志：`/tmp/keytao-s62-baseline-0176089.5HbDgY/baseline-two-classes.log:170`。HEAD 的 `orchestrator.py:2582` 已提前确定性拒绝 missing executionVerb，旧测试却等待第二/三轮 fake 模型回复。仅更新这三个旧断言为一轮 fake 模型、零 sink、未消费后续 fixture、明示拒绝原因。没有为通过旧测试改回更晚的拒绝行为。S57 一处旧“多个同音前缀必须拒绝”的断言按本轮 D 改为人工封签，并保留错误前缀反例。

## 预算与交付边界

**Paid model calls: 0. Real-provider calls: 0.** 测试中的 fake-model 调用只是本地预置响应。
未读取 `.e2e_key`、`.env` 或生产密钥；未运行 `e2e.run`、真实 provider、`pnpm test`；未访问生产、提交、推送或部署。
代码和报告仅在本地工作树修改。以上是静态检查及离线执行证据，不代表生产验证。
