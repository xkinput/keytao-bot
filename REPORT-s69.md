# S69：飞键候选、活动票据改选与发布状态回执

状态：三个缺陷的本地修复及最终离线验收已完成，全部通过。未提交、未推送、未部署。

## 范围与基线

- bot：`2fe52c9 fix: a restrictive adverb does not cancel a selection, and batch-add can be bridged`，起始工作区干净。
- next：`11de2d5 fix: accept a null remark in pull-request validation gates`，已有 `pnpm-workspace.yaml` 用户修改，保留。本轮没有修改 next。
- 仅离线 fixture / fake-tool 测试；不读取生产密钥、不访问生产 API 或主机、不调用真实模型。
- 不运行 `pnpm test`，不提交、不推送、不部署。
- 下文 bot 路径默认相对此仓；next 路径相对 `/Users/rea/code/keytao-org/keytao-next`。明确标记 `HEAD` 的行号对应修复前 `2fe52c9`，其余为本次工作区。

## 根因与修复

### 1. FIXED：折煞的第二读音丢失飞键，默认读音的否定被扩大为全词否定

**确定的丢失点是 bot 为第二读音补码时只建标准链。** 修复前 `HEAD:keytao_bot/utils/keytao_encoding.py:482-485` 只调用 `build_phrase_code_chain` 并清洗结果，因此 `zhé shā` 得到 `qees/qeesi/qeesiu`，没有生成 `fees/feesi/feesiu`。这不是 Next 不支持 `zh + e`。

另外两个点使缺失持续进入回复：

- 修复前 `HEAD:keytao_bot/utils/keytao_review.py:2755` 在返回读音分组时只取 `codes`，没有把同读音 `altCodes` 合并进去。`:1851` 虽把 `altCodes` 加入全局候选，却不能补足读音分组；`:4081-4108` 的另一处读取仅是“恰好一个非默认读音且没有 scoped alternate”的旧兜底，并不是通用飞键合并。
- Next 的 `app/api/bot/phrases/encode/route.ts:33-43` 对当次 `encodePhrase` 的结果调用 `analyzeRequestedCode`；`lib/services/keytaoEncoder.ts:732-793` 只检查这个结果的标准链和飞键链。默认 `shé shā` 的 `unsupported` 只说明这一读音没有 `fees`。修复前 bot `HEAD:keytao_bot/skills/keytao-lookup/tools.py:231-235` 原样透传它，未按第二读音重新核验。

本地纯编码器探针 `e2e/s69_encoder.cjs:33` 直接加载当前 Next 编码器源码，用同一组 fixture 字形、分别指定两种读音，断言：

| 输入读音 | 标准链 | 飞键链 | 对 `fees` 的判断 |
|---|---|---|---|
| `shé shā` | `eees / eeesi / eeesiu` | 空 | `unsupported` |
| `zhé shā` | `qees / qeesi / qeesiu` | `fees / feesi / feesiu` | `flyKey`, `supported: true` |

Next 规则证据：`lib/services/keytaoEncoder.ts:82`、`:636`、`:826-829`；本地 docs 的 `../keytao-docs/guide/advance-in-xkjd/alt-code.md:9-15` 与其一致。探针禁用网络、缓存和读音提供器，只运行纯函数；形码 `iu/uu` 是 fixture，未查询生产或在线字典。因此这证明当前源码的丢失链路能复现用户症状，未声称取得事故当时的 HTTP 原始响应或模型轨迹。

修复后：

- `keytao_bot/utils/keytao_encoding.py:230` 集中枚举 `zh(ai/ao/e): q/f`、`ch(ao/e): j/w`、`uang: m/x` 三类规则，按一字、二字、三字、四字及以上的实际音码取位组合；`:258` 仅沿用受信链上已有的形码后缀，不凭空补形码。
- 单字其他读音及词组其他读音分别在 `:479`、`:531` 展开飞键；`keytao_bot/utils/keytao_review.py:2755` 将规则可归属的同读音 `altCodes` 合入分组，`:2846`、`:4120` 展开各读音链，`:4179` 标记 `flyKey`。
- `keytao_bot/plugins/chat_render.py:1136`、`:1366` 显示“飞键”；`keytao_bot/harness/tools.py:105` 的候选投影保留该标记。真实消息阶段与真实 ToolExecutor 接 fake transport 的 fixture 验证了最终交付和保存的编号一致。
- **「zhe 默认 fe，qe 备选」在 `keytao_bot/utils/keytao_candidate_preference.py:6` 的 `order_reading_codes` 应用**，由 `expand_fly_key_codes` 调用。按去声调后的读音排序，不依赖“哲”这个字。原 `apply_zhe_fe_chain_preference` 仅匹配“哲”的兼容行为保留。

折煞第二读音现在依次为 `4. fees（飞键）`、`5. feesi（飞键）`、`6. feesiu（飞键）`、`7. qees`、`8. qeesi`、`9. qeesiu`；已有/空位信息仍来自占码查询，两个链均保留。

### 2. FIXED：先按方案核验用户指定码，再决定接受、人工复核或拒绝

规则实现位于 `keytao_bot/utils/explicit_code.py:157-222` 和 `keytao_bot/skills/keytao-lookup/tools.py:252-291`：

1. 在完整的已审候选记录上，检查所有已审读音允许的音码前缀。没有出现在候选数组中，不构成非法证据。
2. 合法且形码可由已有链对应的飞键接受为 `flyKey`；合法音码下未核验的形码接受为 `sameSeries`，经 S56 的 `needs_manual_review`、读音与候选 capability 封存，交管理员复核。未审形码既不认定正确，也不拒绝用户申请。
3. 非小写字母、超过六位、与已审读音音码冲突等客观错误仍拒绝，并说出具体原因。例如 `zzzz` 返回“音码前缀不符”及已审前缀。
4. 缺少完整读音、读音来源不可用、语义读音尚未确定时返回 `unverified` 和“未能核验”，不会使用“不在候选中”来断言“不合法”。

`keytao_bot/plugins/chat_commands.py:11988` 还确定性处理当前审词上下文中的“应该是 fees 这个飞键”。这只是核验，不授权写入；回复明确符合 `zhé shā` 规则，需要时提示复核，并说“本次仅核验，未写入”。后续明确加词才走原 S56 占位检查和封存路径。`openai_chat.py:3810` 在通用模型之前接入，因此该回放不会被模型重新误判。

**已有词的维护能力：可以发现缺失飞键。** `test_s69_fly_ticket.py:60` 在“折煞已有 qees、fees 尚空”的 fixture 中直接走审词流程，验证仍列出 `fees` 及其空位状态。审词工具可以服务单词维护；普通已收录词的快捷查词在 `chat_commands.py:3670-3720` 仍提前返回已有词信息，不会自动升级为审词。维护时需进入审词流程；本轮没有批量扫描或回填。

### 3. FIXED：活动确认票据上的编号改选替换预览

旧流程把 `PendingAddWord` 转为服务端 `PendingToolConfirm` 后只保留执行计划，未保存原编号候选的完整快照；编号路径只认识原加词候选或多词审词状态。确认票据上的 `2` 没有可以绑定的原候选，于是落入通用对话/旧票据展示。修复前端到端 fake 回放实际触发了被禁止的模型构造，证明没有走确定性改选路径；没有进行真实模型调用。

- `chat_commands.py:6874-6882` 在首次生成服务端预览票据时保存 `_candidate_origin`；`:12258` 快照包含原词集合、完整候选及读音/占位信息。
- `chat_commands.py:12273` 的 `try_reselect_live_ticket` 用原编号重建选择。震感 `2` 绑定 `qngfv`，生成新的 `create` 预览，丢弃原重排计划，不移动“真敢”。`:12316` 强制新的确认，禁止沿用旧警告自动确认。
- `openai_chat.py:3818` 在模型和通用意图分派之前处理；`chat_commands.py:12379` 的直接 pending 入口也处理同一路径。
- `_submit_after` 保留“加入并提交”的意图；新方案仍需确认。确认后只能写所选 `qngfv`。取消照常清除票据；预览失败保留原票据，并明确“改选预览未完成”，不会声称改选成功。
- 仍拒绝对多词原列表猜编号；过期、其他用户、引述、否定、重复、越界及缺失完整快照均不能执行。执行声明使用原有 `begin_execution`/`abort_execution`。`harness/state.py:610` 在调用工具前剔除 `_candidate_origin`，它不成为 API 参数。

### 4. FIXED / REBUTTED：回执补齐发布状态；不是把旧词条立刻重置为草稿

回执缺信息属实，已补一行：

> 发布状态：「衣品」、「一品」的本次修改现为草稿（未发布）。

`keytao_bot/utils/draft_receipts.py:95-113` 根据实际 `writtenItems/updatedItems` 中有 ID 的修改/删除行列出词名；不会从模型文案推断变更。`:135-165` 合并同轮提交状态，区分草稿、待审核、已入库尚未发布；`chat_render.py:1932-1949` 保留状态来源并替换旧状态行，避免提交后还称草稿。

对“原已发布词条被直接改成 unpublished”的判断，当前源码证据不支持：

- Next `prisma/schema.prisma:47-53` 的 `PhraseStatus` 只有 `Finish/Draft/Reject`；`Published` 属于 `BatchStatus`（`:77-87`），不能混称为词条同一个状态字段。
- `app/api/bot/pull-requests/batch-draft/route.ts:410-452` 建立 `Batch(Draft)` 和 `PullRequest`；PR 默认 `Pending`（`prisma/schema.prisma:317`）。目标 Phrase 在此阶段只读并绑定 fingerprint，没有把目标 Phrase 改为 Draft。
- 审核通过后，`lib/services/batchApprovalService.ts:478-492` 创建的新 Phrase 为 `Finish`；`:498-517` 的 Change 更新不重置状态；`:527-540` 的 Delete 才删除目标行。移码的删除/新增通过这一批次审核流程生效。

所以，对当前草稿操作而言，“新修改尚未发布”是既有审核/发布模型的必经状态，旧词库在草稿创建时仍保持原状。让**旧版本**保持原发布结果，当前流程已经如此；让**新修改**跳过未发布阶段，就需要改变审核与发布流程或增加版本化发布状态，不能只保留一个状态值假装已经发布。本轮只改回执，没有修改发布行为或 Next。未访问生产，不能断言事故中具体批次之后是否已审批或同步发布。

## 修复前复现（非最终验收）

- 最早针对 `FlyCodeTests.test_missing_fly_code_is_valid_for_second_reviewed_reading` 的单测失败：`fees` 被拒绝，原因“音码前缀不符；已审读音对应 eees / qees”。随后才加入方案级前缀核验。
- `/tmp/keytao-s69/red/m-unittest-test_s69_fly_ticket.LiveTicketTests.log`：live ticket 的 `2` 触发 `AssertionError: General model reached`；fixture 构造器立即抛错，真实模型调用为零。尾部 `Ran 1 test in 0.066s` / `FAILED (failures=1)`。
- `/tmp/keytao-s69/red/m-unittest-test_s69_fly_ticket.log`：发布回执断言 `AssertionError: '衣品' not found in ''`。该轮还含一个 fixture 引用错误，随后修正了测试导入；不把该引用错误算作产品缺陷证据。
- 首轮六套的归档在 `/tmp/keytao-s69/first-six/`。状态机有两个旧预期失败（要求丢掉合法 `uang` 飞键；要求不完整、语义不明的读音原样保留肯定结论），审词门禁有一个旧预期失败（`chē` 只应有 j 链）。已按本次明确要求更新断言，同时保留正反边界。
- S56 的旧 `qeskio` 拒绝断言改为规则允许且需人工复核，仍拒绝 `zeskio`。另一个 S56 回执断言补上既有 S68 的未选候选提示；用 AST 比较确认 `_handle_pending_add_word` 与 `_execute_pending_add_word` 相对 HEAD 未改，未修改该行为来适配测试。

## 最终验收

全部命令退出码 **0**。本节只记录冻结代码后的最终运行；不计入上面的红测或首轮结果。

验收归档：`/tmp/keytao-s69/final-logs/`，逐条命令、耗时、退出码及原始日志路径见 `results.json`。驱动脚本 `/tmp/keytao-s69/run_acceptance.py` 只编排仓库自带 offline runners，逐次复制原始日志。`source-before.json` 与 `source-after.json` 的代码/fixture SHA-256 完全一致。最终报告编辑不改变被测源码。

### 六套离线套件与安全套件

实际入口：

```sh
.venv/bin/python e2e/offline_checks.py
```

该入口在七个独立、无凭据且带 I/O guard 的子进程执行。以下为各日志的准确结果尾部：

`test_state_machine.py.log`

```text
============================================================
Results: 2023/2023 passed, 0 failed
✅ ALL TESTS PASSED
============================================================
```

`test_memory_safety.py.log`

```text
----------------------------------------------------------------------
Ran 407 tests in 209.890s

OK
```

`test_security_fixes.py.log`

```text
============================================================
Results: 268/268 passed, 0 failed
============================================================
```

`test_review_gate.py.log`

```text
============================================================
Results: 443/443 passed
✅ ALL TESTS PASSED
```

`test_llm_policy.py.log`

```text
----------------------------------------------------------------------
Ran 11 tests in 0.114s

OK
```

`test_word_discovery.py.log`

```text
============================================================
Results: 290/290 passed
✅ ALL TESTS PASSED
```

`m-e2e.test_safety.log`

```text
----------------------------------------------------------------------
Ran 107 tests in 0.689s

OK
```

### S54 / S58 / S59

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s54_multiword test_s54_renderer test_s54_selection test_s58_bridge test_s58_generic_bridge test_s58_grammar test_s58_advertisement test_s58_explicit_destination test_s59_commonness_evidence test_s59_commonness_route
```

```text
----------------------------------------------------------------------
Ran 117 tests in 0.506s

OK
```

S59 两个模块需要各自独立进程，避免与状态机 fixture 的模块替身交叉污染：

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s59_tool_prompt
```

```text
----------------------------------------------------------------------
Ran 6 tests in 0.082s

OK
```

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s59_general_cap
```

```text
----------------------------------------------------------------------
Ran 11 tests in 0.115s

OK
```

### S62–S68

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s62_selection test_s62_incident test_s62_adversarial test_s62_stale_draft test_s62_reading_resolution test_s62_draft_flow test_s62_commonness
```

```text
----------------------------------------------------------------------
Ran 51 tests in 0.311s

OK
```

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s63_fresh_selection test_s63_general_reply test_s64_existing_query test_s64_explicit_submit test_s65_commonness_render test_s66_readings test_s67_ranking test_s68_selection
```

```text
----------------------------------------------------------------------
Ran 95 tests in 1.601s

OK
```

S63 BCC 三模块使用仓库自己的入口，实际由驱动脚本以全新环境执行（仅 `PATH/LANG/TMPDIR/PYTHONUNBUFFERED`）。对应独立重放命令：

```sh
env -i PATH=/Users/rea/code/keytao-org/keytao-bot/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin LANG=en_US.UTF-8 TMPDIR=/tmp PYTHONUNBUFFERED=1 /Users/rea/code/keytao-org/keytao-bot/.venv/bin/python -m e2e.s63
```

`s63-bcc.log` 中三个子进程的准确结果尾部依次为：

```text
----------------------------------------------------------------------
Ran 22 tests in 0.186s

OK
```

```text
----------------------------------------------------------------------
Ran 10 tests in 1.321s

OK
```

```text
----------------------------------------------------------------------
Ran 3 tests in 0.033s

OK
{"scenario": "S63", "mode": "fixtures/fake-tools", "paidModelCalls": 0, "realProviderCalls": 0, "passed": true, "checks": [{"module": "test_s63_bcc", "exit": 0}, {"module": "test_s63_bcc_ingest", "exit": 0}, {"module": "test_s63_bcc_delivery", "exit": 0}]}
```

### S69 与受影响的 S56 / S60

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s69_fly_ticket
```

```text
----------------------------------------------------------------------
Ran 15 tests in 0.126s

OK
```

这 15 项包含：折煞双链且 fe 优先、真实交付编号与标记、已有词缺飞键、默认拒绝跨读音复核、未知与非法区分、三类飞键及多字取位、缺码 S56 封存、live ticket 改选/确认/提交/取消、失败预览保票据、多词/过期/其他用户/引述边界、发布状态及同轮提交状态。

```sh
.venv/bin/python e2e/offline_checks.py -m unittest test_s56_explicit_code test_s56_security test_s56_e2e_closure test_s60_receipt test_s60_finalizers
```

```text
----------------------------------------------------------------------
Ran 40 tests in 0.108s

OK
```

### Next 本地纯编码器与静态检查

实际以空环境执行（本次 Node 路径保存在 `results.json`）：

```sh
env -i /Users/rea/.local/state/fnm_multishells/42375_1790090855307/bin/node e2e/s69_encoder.cjs
git diff --check
```

两者退出码均为 0。`pure-encoder.log` 保存完整 JSON；三项断言通过：默认读音返回 `unsupported`、`zhé` 读音返回 `flyKey`、`zhé` 的 `altCodes` 含 `fees`。这只验证本地源码与 fixture，不等同线上编码 API 验证。

验收驱动最终输出：

```text
{"sourceUnchanged": true, "allPassed": true}
```

## 预算与交付边界

付费模型调用 **0**，真实 provider 调用 **0**，生产密钥读取 **0**；生产 KeyTao API/生产主机访问 **0**。所有工具结果为 fixture/fake transport；Python offline runner 清空凭据环境并阻断网络和真实凭据文件，S63 使用其独立离线入口；本地 Node 探针不调用读音提供器。

交付仅为 bot 工作区修改；next 原有用户改动保留。没有 commit、push、deploy 或生产验证。如以后两仓都需要部署，既定次序仍为 bot 先、next 后；本轮未修改 next，也未部署任何仓库。
