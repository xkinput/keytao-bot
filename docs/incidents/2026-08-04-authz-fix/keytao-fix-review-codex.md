# Codex 跨模型评审报告（2026-08-05）

- 评审者：Codex（GPT，codex-cli 0.145.0），与实现方（Claude Opus）零共享上下文
- 评审对象：keytao-bot / keytao-next 两仓全部未提交工作区改动（2026-08-04 授权修复批次）
- Codex session：`019fcde2-1fab-7d71-bff8-446f971f9119`（`codex resume` 可回放完整推理）
- 本文件由主会话从 Codex 最终结论转录归档（Codex 沙箱只读，无法自行写文件）

## 结论：HOLD，不可提交或按 bot → next 部署

## 阻断发现

### P0-1 尾置「记录/转述」框架可直接授权删除
例如「请把这句删除草稿条目 12 记录下来」被判定为授权，并实际调用 `pr_id=12`。
位置：`keytao_bot/harness/tools.py:292`

### P1-1 补救建议未校验全部操作数
补救建议只验证词条，未验证 `target_code/code/type/action`，可生成
`@我 顺延「吃席」到 zzzz` 等夹带参数的指令。
位置：`keytao_bot/harness/tools.py:621`

### P1-2 A6 未闭环（双票与 absence anchor 覆盖）
真实 Next 会再次返回 warning challenge，shift 仍需两张票；无草稿时 provisional UUID
覆盖 absence anchor，确认必然 stale。
位置：`keytao_bot/harness/orchestrator.py:1200`、`keytao-next/app/api/bot/pull-requests/batch/route.ts:329`

### P1-3 recall 后旧票未跨会话作废
工具确认路径完成 recall 后，未作废同一操作者其他会话中的关联旧票；submit/recall
又不推进版本，旧授权可在 Draft → Submitted → Draft 后复活。
位置：`keytao_bot/plugins/openai_chat.py:5832`

### P1-4 部署中间态仍可重现影子空批次
bot-first 只是协议兼容，中间态仍会被旧 Next 的读时建批次重现影子空批次；必须隔离
两步部署间的流量。反向 next-first 会让旧 Bot 的首次 add/shift 直接失败。
（属部署流程问题：需在 bot 发布与 next 发布之间停 bot 流量，而非代码修改。）

## 验证限制声明
- Next `tsc --noEmit` 与两仓 `git diff --check` 通过。
- **测试未能运行**：Vitest 因沙箱 `EPERM` 为 0 tests，Bot 套件在临时目录/SQLite
  初始化阶段退出。以上发现均为静态分析结论，未经动态验证。
- 无联网、无数据库变更、无源码改动。

## 独立验证结果（Claude Opus 验证员，2026-08-05；读代码 + 只读 venv 跑仓库自带校验器，两仓 bot 套件 130 tests / 1195 checks 通过，未改任何源码、未跑 pnpm test）

**总判定：本批次不构成安全回归。四项发现无一是「未确认即写入」或「跨用户越权」。唯一应拦部署的是 P1-2（A6 未闭环 + 无草稿顺延永久死路），且它疑似由本批新锚定逻辑引入。**

### P0-1 —— CONFIRMED，但严重度被高估（实际 P2，非 P0）
- `message_authorizes_mutation("请把这句删除草稿条目 12 记录下来")` 返回 `True`。逃逸点在 `tools.py:525`：动词开头校验失败后，`_EXPLICIT_REQUEST_PREFIX_RE` 命中开头「请」照样授权；转述过滤 `_DATA_CONTEXT_RE`（tools.py:292）是 `^` 锚定，只挡**开头**的「记录/他说」，**尾置**「把…记录下来」漏网。
- **但不产生未确认写入**：`keytao_remove_draft_item` 返回 `requiresConfirmation` 且无 `expected_target_digest`（skills/keytao-draft/tools.py:2888-2895），而 Next 端每个未确认请求现在都是只读预览（batch/route.ts:332）。所以最坏结果只是「弹一次确认框 + 存一张票」，真正写入还需第二次「确认」。且 self-scoped（platform_id 就是发送者，无跨用户）。
- 反例控制组全部正确返回 `False`：`他说删除草稿条目 12`、`记录一下：删除…`、`请把「删除…」记录下来`、`朝歌说要删除…先记下来`。

### P1-1 —— 部分 CONFIRMED：`code/target_code` 可夹带；`type`/`action` 为**误报**
- `_suggestion_operands`（tools.py:621-649）只返回 word；`code/target_code` 确可夹带：`顺延吃席`+`target_code='zzzz'` → 生成 `@我 顺延「吃席」到 zzzz`。自检 tools.py:775-791 是循环校验（拿建议串校验建议串）。
- **误报**：`action` 由 `_tool_intent_pattern`（tools.py:559-582）绑定用户原文动词，夹带 `action='Delete'` 得 `''`；`type` 在 `canonicalize_arguments`（tools.py:1389-1412）已被剥离。严重度 P2——用户仍须真的把命令发出去，且执行绑定 `_code_is_bound_to_target`（tools.py:1817）仍在。

### P1-2 —— CONFIRMED（两半都成立），**真正的部署拦截项**
- **两票**：orchestrator 自动确认只传 `{confirmed_plan_digest,batch_id,expected_content_version}`（orchestrator.py:1248-1251），漏 `warningDigest`，于是顺延内部写 `confirmed=False`（tools.py:3887）→ route.ts:332 判为预览；票 1 重复同一未确认写，只有票 2（openai_chat.py:5478-5480）带 warningDigest。一条指令 → 两次用户确认。**违背 A6 验收标准「一次确认完成顺延」。**
- **无草稿路径永久死路**：预览返回 `batchId = targetBatchId ?? randomUUID()`（route.ts:299,337），两个票据构造器都把这个 provisional UUID 原样当 CAS 锚（orchestrator.py:1204-1221；openai_chat.py:5442）。回放时 `latest-draft/items` 对未知 id 返回 `batchId:null`（items/route.ts:71-82）→ `current_batch_id=""` → 守卫 tools.py:3868 判定 `staleConfirmation`，**每次都失败**。绿测 `test_shift_without_any_draft_executes_in_one_authorization`（test_memory_safety.py:3447）是 mock 产物，把 confirm 桩成 `{"success":True}`，真实 Next 永不返回。**这条最接近原事故（朝歌反复被挡）。**

### P1-3 —— CONFIRMED，但**先于本批已存在，非本批回归**
- `invalidate_actor_related` 只有一个调用点：自然语言 recall 命令（openai_chat.py:7248）。工具确认完成路径（openai_chat.py:5832-5853）与整个 orchestrator 路径都不调它。且 recall 写 `status:'Draft'`、submit 写 `status:'Submitted'` 均不推进 contentVersion → Draft→Submitted→Draft 后旧票仍匹配。窗口受 2h 票据 TTL（state.py:306）限。严重度 P2，`invalidate_actor_related` 无任何测试覆盖。

### P1-4（部署中间态）
以部署流程消解：停 bot → 发 bot → 发 next → 起 bot（消灭中间态流量），无需改代码。

## 结论与建议
- **安全上可发**：无未确认写入、无跨用户越权。
- **唯一应拦**：P1-2 无草稿顺延永久死路（疑似本批新锚定逻辑引入的回归）+ A6 两票未闭环。修法：bot 端不要把 provisional batch id 当 CAS 锚，或 re-read 时接受 `batchId==""`。
- P0-1 / P1-1 / P1-3 均降级 P2，进 backlog 下一轮修。
