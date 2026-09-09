# S60 — 完整写入回执与批次链接

基线：`main`，`f81a815f738b2702ec050c19c7114f292e805414`，初始工作树干净。
本轮只在本地修改和验证，未暂存、提交、推送、部署或访问生产。
**付费模型调用 0；真实 provider 调用 0。未读取生产密钥、真实 `.env` 或 `.e2e_key` 内容；未运行 `pnpm test` 或 `e2e.run`。**

## A — 漏项原因与实际回执

确认事故中的回执缺陷，不反驳“写入完成、回执不完整”的诊断。生产事实来自本轮给定的 transcript/log；没有重新访问生产核验。

基线 `chat_commands.py:6188` 的 `_format_ranked_shift_success` 从 `shiftPlan.word/targetCode`、
`shifted`、`draftUpdates` 和分步摘要取展示内容。合并计划的普通 Create 虽在 `shiftPlan.items`
中执行，却不在这些摘要投影中，因此“沃集鲜”消失。编排器原 `_successful_write_receipt`
也从请求参数和顺延摘要投影，shift 分支没有普通新增行；compact 草稿列表还只展示前五项。

本轮不把完整计划或整批草稿直接当作本轮写入。旧本地 S38 artifact 的真实工具响应只有
`pullRequestCount/successCount/draft_snapshot/shiftPlan`，没有逐项创建回执；`draft_snapshot`
和普通 batch 的 `draftItems` 均可能包含旧草稿。

现在复用 `completed_draft_undo.py:279` 操作日志已有的、每次有效工具调用的局部 before/after
快照，在 `:233` 附加 `writtenItems`（新增 PR ID）及 `updatedItems`（既有 PR 的实际修改）。
不使用整轮累计的 `capture.before` 计算单次增量，故“写入、再写入、再提交”不会重复计数。
提交返回空 PR 增量，保留此前写入事实；提交后不读草稿。正常成功路径没有新增查询。
`partialWrite` 保留失败状态，并通过相同快照检查取得已完成子集，不把计划剩余项算成成功。

`completed_draft_undo.py:177` 绑定批次、实际写回版本、可用的 PR 数量与请求身份。
没有写回版本时，只接受数量一致、请求匹配且旧行未变的新增差集。
跨批次、后置快照混入并发写入、数量不符或快照缺失时，明细标为未核验，不从计划补造成功项。
服务端新增的审核备注和人工审核标记保留为实际回执字段，不因正常的审核补充而隐藏该 PR。
没有 operation capture 的调用只能保留完整且 ID 合法的工具逐项回执，否则同样标明未核验。

`draft_receipts.py:15` 是共享逐词投影。一个 Delete/Create 移码对合并展示，但保留两个 PR ID
的覆盖关系；创建、词文本 Change、权重 Change 和既有草稿权重更新分别按实际内容展示。
`receipt_change_lines` 保留失败／跳过原因；原因没有返回时明确说尚未确认。
`noWrite`、已应用及重放不计入本轮新增，未重复写入的原因只归属于对应词，不传播到另一笔未知结果。

**计数解释：本例 4 条 PR 对应 3 个词级变更，不可能同时要求 4 个可见词组。**
不变量检查展开每个 `WrittenChange.created_ids` 后，数量与 ID 集合都严格等于本轮创建的 PR；
本例同时断言 4 个覆盖 ID、3 个展示词、掉岗移码覆盖 2 个 ID。既有 PR 更新单独计数。

原始 `_query_words` 通过内部回执顺序字段保留到确认重放；S57 沿用 `listed_words`。
显示按用户词序排列，被挤出的词紧跟其对应的新词。该字段不发送给写工具、不改变授权或计划摘要。

## B — 链接消失的直接原因

基线 `continue_with_submit_preview` 拼接两个成功回执后调用 `_dedupe_authoritative_link_lines`。
该函数实际会删除所有 `/batch/` URL，而不仅仅删除重复 URL（现 `chat_render.py:1637`）。
因此即使 shift 和 submit 都带了可信链接，组合路径仍会把它们全部删掉。
另一个边界是 submit 成功模板只读 `batchUrl`，只返回精确 `batchId` 时也会没有链接。

新共享 `chat_render.py:1932` 的 `finalize_draft_receipt` 从可信响应重建完整变更与链接。
组合路径传入 submit 和原 write 两份可信响应；普通 draft formatter 也传入 list fallback，
既保留精确身份，又不丢已有的可信链接。已有 localhost fixture URL 在交付前保留。
只有精确、非 provisional 的批次身份才可补成链接，absence-CAS 临时身份仍显示“待确认后生成”。

最终交付 `openai_chat.py:2693` 还按当前 actor 的本轮工具回执恢复条目、提交状态和链接。
摘要异常／被拒绝时，已有写入不会再被描述成“本次未写入”；已提审或自动入库也按实际 submit
回执恢复。异 actor 不能取得该回执。每个实际批次保留一次链接，最后沿用原平台映射：

- QQ：`https://keytao.rea.ink/batch/<id>`
- Telegram：`https://keytao.vercel.app/batch/<id>`

## C — 收尾路径 sweep

| 路径 | 本轮处理与验证 |
|---|---|
| 多词 shift／eviction 加普通 Create | `_format_ranked_shift_success` 优先使用实际增量；1、3、4 PR 及混合回执覆盖 |
| 普通 Create／batch compact 列表 | 完整本轮变更另由共享 finalizer 产生，不依赖全草稿前五项摘要 |
| S57 权重重排加 Create | Change 和 companion Create 共用覆盖不变量；既有 PR 更新不冒充新建 PR |
| S57 多步、部分完成 | 保留最后成功的 mutation ack 版本和创建数量，不能把后置读取版本当写入确认 |
| 编排器正常结束、提交失败、传输失败、迭代耗尽 | `_successful_write_receipt`、`_receipt_completion_reply`、`_finalize_reply` 优先使用逐项真实回执 |
| 摘要被交付边界替换 | 捕获 actor 内的全部增量；恢复变更、Submitted／autoApproved 状态及链接 |
| noWrite、重放、缺失明细、并发漂移 | 不制造新写入；逐词保留已知原因或未确认状态 |

现有“删除草稿 PR”和“撤回提审”继续保留原来的删除／撤回回执，不能把它们误当成空的新增 PR 回执。
本轮没有改变写入目标、确认资格、服务端接口、审批流程或持久化 schema。

## S60 fixture 与反馈循环

`test_s60_scenario.py` 通过真实多词查询路由及 pending executor 重放：
`吊杠 沃集鲜` → `加入并提交`。审词及工具写回均为 fixture；两个意图模型入口被设置为调用即失败。
查询先保存一个可信 reviewed record，再渲染；确认绑定原计划、版本和摘要，fake 实际创建 4 条 PR
并完成 submit。QQ／Telegram 各跑原序和逆序，共 4 项。没有将 S60 加入 real-provider scenario pack。

QQ 原序最终 fixture 回执：

```text
✅ 操作已完成
已变更：吊杠 → dcgp、掉岗 dcgp→dcgpi；沃集鲜 → wjxa

✅ 批次已提交审核。

草稿地址：https://keytao.rea.ink/batch/s60-e4d21c4d
```

原始红灯保存在 `/tmp/keytao-s60-offline/s60-scenario-red.log`：4 个场景均已创建并提交，
但实际旧回执缺“沃集鲜”和链接。最小单元复现也先红：漏词一项，以及 QQ／Telegram 缺链接两项。

独立同模型审查发现并修复了并发快照归属、原词序被计划顺序覆盖、noWrite 丢词、提交状态兜底遗漏、
noWrite 原因扩散等边界；每项均有新增离线回归。不称为跨模型验证。

首轮七项完整记录保存在 `/tmp/keytao-s60-offline/first-suites/`：state 为 2022/2023，
唯一失败是共享 formatter 未继续传 list fallback 链接；已补回，原断言不变。
memory 为 407 tests、12 failures；其他五项 exit 0。

12 个 memory 测试的假写工具原来只有成功标志和计划，无法证明逐项实际写入。
现在为它们补入字面的整数 PR ID 与实际 Create／Delete／Change 行，不从计划自动生成断言。
模型填充文案的断言改为实际变更与精确链接，原工具调用、授权、摘要、审核标志和 pending 消费检查保留。
同码前插仍明确核对成功回执的 Create 100／Change 101 权重。12 个原失败方法逐项复跑均通过。

## 零付费离线启动器与验证记录

`e2e/offline_checks.py` 启动独立 Python 子进程，只传固定、无凭据的环境；禁止 socket 联网、
非受保护子进程和真实密钥文件读取。真实 dotenv 返回空值；本测试进程新建的临时 fixture 配置可用。
启动器自身 3 项保护检查通过。先前临时位置已移入 `e2e/`，S60 不依赖未交付的临时文件。

复跑命令：

```sh
.venv/bin/python e2e/offline_checks.py
.venv/bin/python e2e/offline_checks.py -m e2e.s60
.venv/bin/python e2e/offline_checks.py -m unittest test_s54_selection test_s57_reorder test_s56_undo test_s56_undo_review
```

第二条已通过 45 项：场景 4、收尾 6、实际增量 20、展示不变量 15；各模块独立进程 exit 0。
第三条已通过 55 项，`Ran 55 tests in 0.407s`，`OK`。

最终第一条在受保护的独立子进程中依次执行以下七项原脚本／模块，全部 exit 0。
下表保留实际尾部；耗时为启动器测得的各子进程总耗时，不混同 unittest 的内部计时。

| 检查 | 实际尾部 | 子进程耗时 |
|---|---|---:|
| `.venv/bin/python test_state_machine.py` | `Results: 2023/2023 passed, 0 failed` / `ALL TESTS PASSED` | 1.391s |
| `.venv/bin/python test_memory_safety.py` | `Ran 407 tests in 208.379s` / `OK` | 209.152s |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed` | 0.762s |
| `.venv/bin/python test_review_gate.py` | `Results: 443/443 passed` / `ALL TESTS PASSED` | 23.151s |
| `.venv/bin/python test_llm_policy.py` | `Ran 11 tests in 0.099s` / `OK` | 0.438s |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed` / `ALL TESTS PASSED` | 0.252s |
| `.venv/bin/python -m e2e.test_safety` | `Ran 107 tests in 0.622s` / `OK` | 0.881s |

完整日志、最终七项结果和指纹核对分别位于 `/tmp/keytao-s60-offline/`、
`/tmp/keytao-s60-offline/results.json`、`/tmp/keytao-s60-offline/completion-verification.json`。
S60 汇总尾部为 `scenario=S60, mode=fixtures/fake-model, paidModelCalls=0,
realProviderCalls=0, passed=true`，四个测试模块均 exit 0。

最终测试使用的 14 个 Python 修改／新增文件均已冻结并逐一复核 SHA-256：
`changedSinceFreeze=[]`、`missingFromFreeze=[]`。测试结束后只补齐本报告。
`git diff --check` exit 0，暂存路径为空，HEAD 仍为
`f81a815f738b2702ec050c19c7114f292e805414`。工作树共 14 个 Python 文件和本报告，未提交。

验证范围仅为静态源码、离线 fake 工具／模型与规定测试；没有真实 provider、线上用户、生产服务或生产数据验证。
**付费模型调用 0，真实 provider 调用 0，生产访问 0；所有改动仅在工作树中。**
