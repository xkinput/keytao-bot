# keytao 修复交叉评审报告（独立评审者，零共享上下文）

评审对象：`/Users/rea/code/keytao-org/keytao-bot` 与 `/Users/rea/code/keytao-org/keytao-next`
两个仓库的工作区未提交改动（bot HEAD `e5dfd27`，next HEAD `1c2b4e3`，均未 commit）。
评审方式：只读代码审查 + 在 `keytao-bot/.venv` 内对真实函数做对抗性探针 + 4 次变异检验。
**两个仓库在评审前后逐字节一致**（`git diff` / `git status --short` 与评审开始时的快照 `diff -q` 全等）。

探针与证据全部在 scratchpad：
`rev_probe1_authz_delta.py`（HEAD vs 工作区授权判定 diff）、
`rev_probe2_suggestion_replay.py`（建议指令新一轮回放）、
`rev_probe3_misc.py`（未声明参数穿透 + 绑定层负例）、
`rev_probe4_word_discovery.py`（真实服务端语义下的 word_discovery 链路）。

---

## 复核过的实现者主张（属实部分）

| 主张 | 结论 |
|---|---|
| `_MUTATION_INTENT_RE` 一字未动 | **属实**。`_MUTATION_INTENT_RE` / `_DELETE_INTENT_RE` / `_NEGATED_MUTATION_RE` / `_PROTECTED_WORD_RE` / `_ACTION_TOKENS` 五个定义与 HEAD `diff` 全等 |
| 测试数字 | **全部复现**：memory_safety 113、state_machine 1186、security_fixes 207、review_gate 150、llm_policy 9、isolation_fixes 67、word_discovery 256；keytao-next `vitest --config vitest.unit.config.ts` 51 文件 / 513 用例全绿，`tsc --noEmit` 无错误，eslint 无告警 |
| suggestedCommand 先过同一套校验器 | **属实且经实测**：7 个工具 9 组参数生成的建议，在**清空 trusted 映射的新一轮上下文**里回放，`message_authorizes_mutation` 与 `_validate_current_message_binding` 全部通过（rev_probe2） |
| A6 第二次调用完整重跑校验 | **属实**：`_auto_confirm_shift_plan` 走 `ToolExecutor.call`（`tools.py:1122-1132`），`_validate_policy` 完整重跑；`confirm_args` 不含 `confirmed` 键，绕不过 `ticket_required` 分支；服务端 CAS（`skills/keytao-draft/tools.py:3856-3867`）未动 |
| 绑定层扩表未误伤负例 | **属实**：`苹果汁顺延到 wkxk` 不能绑 `苹果`、词条外的保护词仍拦截、闲聊带词+码不授权（rev_probe3 part B） |
| B2 选择器 + B1 纯读能闭合事故 | **属实**：`draft-pointer.test.ts` 用内存假 prisma 跑真实路由复现事故序列，`batch.create` 直接抛错；变异检验（去掉 content 优先）立即变红 |

---

## 发现清单

### 【major-1】`_EXPLICIT_REQUEST_PREFIX_RE` 加「执行」把闲聊抬成 AUTHORIZED（实测 8/8 反例）

- 位置：`keytao-bot/keytao_bot/harness/tools.py:273-280`
- 实现者自报偏差的理由是「执行只对**已过授权视图**的子句生效，不可能抬升闲聊」。
  **这个论断是错的**：`_mutation_authorization_view`（`tools.py:471-476`）除了 `is_positive_command`
  之外，还会把 `is_protection_clause` 的子句放进视图；而 `保留` 同时在
  `_PROTECTED_WORD_RE` 和 `_MUTATION_INTENT_RE` 里。于是任何「执行 … 保留 …」形态的
  日常句子都会进视图，再被新加的 `执行` 前缀满足最后一道「动词在 0 位 **或** 命中显式请求前缀」闸门。
- 实测（rev_probe1，HEAD vs 工作区逐句对比）——以下 8 句 **HEAD=False → 工作区=True**：

  ```
  执行结果保留一下 / 执行完保留原样 / 执行保留策略 / 执行日志保留 7 天
  执行前保留一份快照 / 执行的时候保留原来的编码 / 执行方案已经保留在文档里 / 执行摘要保留在群公告
  ```

  同一批语料里，实现者 hazard 扫描点名的那些句子（提前告诉我结果 / 占用率是多少 /
  执行完了记得删除备份 / 他说执行顺延吃席wkxk / 如果执行顺延会怎样 / 别执行顺延 …）确实仍为 False。
  问题出在实现者的 hazard 语料**没有覆盖「执行 + 保护词」这一类**。
- 失败场景：群友闲聊一句「执行结果保留一下」→ `mutations_allowed=True` → 第一道闸门失效 →
  **A5 的只读轮摘工具随之失效，模型在这一轮拿到全部 7 个写工具**。实际写入仍被绑定层挡住
  （submit/recall 走 `re.fullmatch` 的独立指令匹配，不会被误触），所以这是纵深防御回退，
  不是直接写洞——但它正是规格 A2 明文红线要防的东西。
- 建议修法（已实测）：把 `执行` 从 `_EXPLICIT_REQUEST_PREFIX_RE` 撤回，改为在
  `message_authorizes_mutation` 末尾用 `_COMMAND_PREFIX_RE` 先剥前缀再判「动词在 0 位」：

  ```python
  stripped = _COMMAND_PREFIX_RE.sub("", authorization_text, count=1)
  m2 = _MUTATION_INTENT_RE.search(stripped)
  return (m2 is not None and m2.start() == 0) or bool(_EXPLICIT_REQUEST_PREFIX_RE.match(authorization_text)) or ...
  ```

  实测结果：8 条 hazard 消掉 7 条（只剩「执行保留策略」，它与 HEAD 就已为 True 的
  「保留原样」同类，非本次引入），而 A2 全部验收句式 + 13 条必须授权句式**全部保持 True**。
  另请把这 8 句作为负例进 `test_memory_safety.py`。

### 【major-2】`word_discovery` 的「先问后确认」是死代码，链路仍然写不进去（自述称已修，实测未修）

- 位置：`keytao-bot/keytao_bot/utils/word_discovery.py:1680-1712`
- 真实服务端契约：`POST /api/bot/pull-requests/batch-draft` 在 `confirmed !== true` 时返回
  **HTTP 400**（`keytao-next/app/api/bot/pull-requests/batch-draft/route.ts:342-367`），
  而 `http_client.keytao_json`（`keytao_bot/utils/http_client.py:400-407`）对**任何非 2xx 都抛
  `KeytaoApiError`**。因此第一次 `confirmed: False` 的 POST 必然抛异常，
  `if added.get("requiresConfirmation")` 这一整段**永远不会执行**。
- 实测（rev_probe4，按真实服务端语义 stub）：

  ```
  calls: GET /api/bot/batches/latest-draft
         POST /api/bot/pull-requests/batch-draft  confirmed=False
         GET  /api/bot/batches/latest-draft/items      ← 异常分支的补偿探测
  result.success = False   message = 写入草稿失败：请确认将 1 个修改写入草稿
  confirmed batch-draft writes reached the server: 0
  ```

- 不是回归（HEAD 的 `confirmed: True` 无 digest 同样 400），但**摘要把它列为已修复是不成立的**。
- 测试为什么没抓到：`test_word_discovery.py:652-659` 断言链路恰好是
  `latest-draft → batch-draft → submit → auto-approve` 四步，stub 直接返回不带
  `requiresConfirmation` 的成功 dict。**只要真按 preview→confirm 实现，这条断言反而会挂**，
  所以现有测试结构上不可能覆盖新分支。
- 建议修法：① 用不抛异常的调用（`keytao_request` + 手动读 body，或给 `keytao_json` 加
  `allow_status={400}`）拿到 preview 的 `contentVersion` / `warningDigest`；② 同批改写
  `test_auto_ingest_chain_and_degradation` 的路径断言为 5 步（batch-draft 出现两次，
  第二次 `confirmed=True` 且带 digest），并加一条「服务端 400+requiresConfirmation 时必须补发确认」的用例。

### 【major-3】两仓库部署顺序被强耦合：keytao-next 先上线会打断所有「草稿已全部提审」用户的加词

- 位置：`keytao-next/app/api/bot/batches/latest-draft/route.ts:46-62`（读不建）×
  `keytao-bot/keytao_bot/skills/keytao-draft/tools.py:1130-1139`（None 透传）
- 失败场景：用户把草稿全部提审后（批次变 `Submitted`，名下**没有** Draft），
  新 keytao-next 的 `GET /latest-draft` 返回 `batchId: null`；**旧版 bot** 的
  `if not batch_id: return {"success": False, "message": "无法获取草稿批次，请稍后重试"}`
  会让「添加词条」直接失败。这是极常见状态，不是边缘用例。
  反向（bot 先上线 + 旧 next）是安全的：旧 next 的 get-or-create 永远不返回 null，None 透传分支不触发。
- 建议：明确记录并执行「**先发 keytao-bot，再发 keytao-next**」，或在 keytao-next 上线时
  保留一次性兼容（例如 `?createIfMissing=1` 参数仅供旧 bot 使用），观察期后移除。

### 【major-4】A5 与 A1 互相抵消：事故第 1 句在只读轮拿不到任何 suggestedCommand

- 位置：`keytao-bot/keytao_bot/harness/orchestrator.py:189-215`
- `mutations_allowed=False` 时全部 `MUTATING_TOOL_NAMES` 被从工具清单摘除，并追加一条
  system 消息让模型「直接向用户说明需要什么样的指令」。suggestedCommand 只在**模型仍然点名
  一个已被摘掉的工具**时才产生（`orchestrator.py:447-465`）。
- 失败场景：事故第 1 句「把吃席的编码放在赤溪前面」实测 `message_authorizes_mutation=False`
  → 只读轮 → 合规的模型不会去调不存在的工具 → **没有任何结构化拦截、没有 suggestedCommand**，
  模型只能自己编一句提示——这正是事故「每次一个新格式」的机器。
  摘要 §2 矩阵第 20/21 行声称该句「拦截输出给出自检可通过的建议指令」，
  只在测试里成立（`ShiftAuthorizationTests._shift` 硬编码 `writes_allowed=True`，
  与生产的只读轮不同路径），生产上不保证。
- 建议：给只读轮补一个**非写**工具（如 `keytao_request_write_authorization(tool, args)`），
  它复用 `self_checked_suggested_command` 返回确定性的建议指令；或在只读轮的 system 消息里
  直接注入本轮可用的指令模板，让「格式」不再由模型自由发挥。

### 【major-5】suggestedCommand 把模型自选参数变成一句「已通过校验器」的现成授权指令

- 位置：`keytao-bot/keytao_bot/harness/tools.py:1259-1275`（`self_checked_suggested_command`）、
  `orchestrator.py:447-465`（只读轮/被摘工具分支）
- `_suggested_command_text` 的输入是**模型提交的参数**，不是用户说过的内容。
  实测（rev_probe3 part B）：用户只是问一句「吃席到底怎么打 wkxk」，
  拦截回包里就带 `suggestedCommand: "@我 顺延「吃席」到 wkxk"`，且拦截文案要求模型
  「原样转述给用户，不要自创格式」。
- 失败场景（注入威胁模型）：记忆/引用里的注入内容让模型提出一个写操作 → 本轮 `writes_allowed=False`
  本来应该到此为止 → 现在 bot 会把一条**验证器保证能通过**的指令递到用户面前，
  并以「这是唯一可行的格式」的口吻要求用户发送。事故里用户正是逐条照做的那类用户。
- 建议：只在 `blockReason == binding_incomplete` **且本轮 `writes_allowed=True`**（即用户已经
  给出了执行意图、只是绑定不全）时给 suggestedCommand；`source_untrusted` /
  只读轮被摘工具这两条路径只说明原因，不给现成指令，或把建议限制为
  「用户本轮原文里已出现过的词条 + 编码」。

### 【minor-6】`keytao_get_batch_preview` 新增的 `batch_id` 可被模型自由指定，打到一个无鉴权公开接口

- 位置：`keytao-bot/keytao_bot/skills/keytao-draft/tools.py:2031-2063`（新增参数）、
  `:4005-4021`（schema 仍是 `properties: {}`）、`keytao-next/app/api/batches/[id]/preview/route.ts:154-172`
- schema 没有声明 `batch_id`，但 `tools.py:58-63` 会剥掉 `additionalProperties: False`、
  `_validate_json_schema` 也只校验已声明字段，未声明参数直达函数。
  实测（rev_probe3 part A）：`keytao_get_batch_preview({"batch_id": "someone-elses-batch-uuid"})`
  的 kwargs 原样送进函数。该函数用 `client.get(url)` **不带任何鉴权头**请求
  `/api/batches/{id}/preview`，而该 Next 路由是完全无鉴权的 `findUnique`。
- 失败场景：注入内容里带一个别人的批次 id，模型被诱导调用 → 别人的草稿 diff 被打印到群里。
  （接口本身早已公开，不算平台新漏洞；新的是「LLM 可控地读任意批次」这个能力。）
- 建议：把 `batch_id` 限制为**本轮已由服务端返回过**的批次（沿用 trusted-context 模式），
  或把预览改走带 bot 鉴权、按 `creatorId` 收敛的接口；同时把参数正式写进 schema（别让它是隐藏参数）。

### 【minor-7】`_deduplicate_block_reason` 只按 reason 去重，跨工具/跨目标误伤

- 位置：`keytao-bot/keytao_bot/harness/orchestrator.py:1109-1128`（`reported_block_reasons` 定义在 `:196`）
- 去重键只有 `blockReason` 字符串，作用域是整轮 run 的全部工具调用。
- 失败场景：用户一句话里两个操作（「添加甲 aa，删除条目 12」），第一个因绑定不全被拦
  → 第二个即使是**另一个工具、另一个目标**、本来应该给出可用建议，也会退化成
  「换写法没有用。请直接回复用户，不要再重试。」并丢掉 `suggestedCommand`。
- 建议：去重键改为 `(blockReason, tool_name, 关键参数指纹)`。

### 【minor-8】顺延在「用户名下完全没有草稿批次」时硬失败（实现者已列为遗留，确认影响面）

- 位置：`keytao-bot/keytao_bot/skills/keytao-draft/tools.py:3776-3821`
- `keytao_shift_phrase_code` 用 `keytao_list_draft_items(batch_id=...)` 取 `existing_draft`；
  B1 之后无草稿时该接口返回 `batchId: null` → `current_batch_id == ""` →
  返回「当前草稿缺少可验证的内容版本，顺延未执行」。
- 影响面：**提审完全部草稿之后**（Draft 归零）就会触发，不只是全新用户；
  而「提审 → 想改一个编码」正是事故场景的相邻状态。可接受为遗留，但请排在下一批。
- 建议：无草稿时把 digest payload 里的 `batchId` 定为 `""`、`contentVersion` 定为 `0`，
  确认写入时把 `batch_id=None` 透传给 `/batch-draft` 由服务端 `batchIsVirtual` 路径落库
  （该路径已存在，见 `batch-draft/route.ts:178-192, 377-383`）。

### 【minor-9】`_keytao_strict_batch_add_to_draft` 放宽的守卫不可达

- 位置：`keytao-bot/keytao_bot/skills/keytao-draft/tools.py:3539-3547`
- 全仓唯一调用点是 `keytao_shift_phrase_code:3869`，永远传非空 `batch_id` + `int` 版本号，
  且更早的 `:3812-3821` 已先行返回。改动为无效果的门面修改，容易让人误以为 minor-8 已缓解。
- 建议：连同 minor-8 一起改，或加注释说明它只是为将来的直接调用者预留。

### 【minor-10】A3 的两半各自只有一条断言在保护，头牌测试对其中一半不敏感

- 变异检验证据（4 次，全部检出，全部已还原）：

  | # | 变异点 | 结果 |
  |---|---|---|
  | 1 | `tools.py:930-941` 邻接过滤回退成 `_span_distance == 0` | FAIL 2 处，但 12 条句式里**只有**「执行顺延吃席wkxk赤溪wkxkv」变红 |
  | 2 | `tools.py:476` `normalized` 改回 `compact`（撤销空白保留） | FAIL 1 处，且**不是** 12 条句式，而是 `test_every_suggested_command_passes_its_own_validator` 的 `keytao_batch_remove_draft_items` 子例 |
  | 3 | `orchestrator.py:218-224` 关掉 `_auto_confirm_shift_plan` | `test_bound_shift_executes_without_asking_for_a_ticket` 变红（`1 != 2`） |
  | 4 | `botDraftBatch.ts:83-87` 去掉「有内容优先」首查 | keytao-next 6 个用例变红（含 draft-pointer 事故回归） |
- 结论：测试**是真语义测试**（不是照抄实现），但 A3 两个修复点的覆盖各只有一根丝。
- 建议：在 `ShiftAuthorizationTests` 里对「顺延 吃席 wkxk」「顺延：吃席 wkxk」这类**带空格**句式
  单独加一条断言 token 边界的用例，让撤销 `normalized` 会直接打红头牌测试。

### 【minor-11】`_has_protection_outside_target` 在「词条本身就是保护词」时会忽略整句的保护语

- 位置：`keytao-bot/keytao_bot/harness/tools.py:756-767` + `:1604-1609`
- 逻辑是「跳过与目标 span 重叠的保护词」。当 `word == "保留"` 时，句中**每一个**「保留」
  都是目标 span，于是「顺延保留到 wkxk，保留原编码」里用户真正的保护意图也被吞掉。
- 影响很小（要求词条恰好是保护词），但既然 A4 的目的就是支持这类词条，值得收紧。
- 建议：只跳过**被工具参数点名的那一处** span（例如引号内的那次出现），
  而不是全部 exact-match span。

### 【note-12】`_LEADING_MENTION_RE` 在生产链路上是冗余的

`openai_chat.py:375-378` 的 `_LEADING_COMMAND_PREFIX_RE` 已经先剥掉 `@\S+`，
所以 `tools.py:311-314` 的新正则在生产上永远匹配不到东西；它真正的作用是让
`self_checked_suggested_command` 自检的那串 `"@我 " + candidate` 能被同一个验证器接受。
无害，但请在注释里写明这一点（现注释说的是「某些适配器会把 @ 留在正文里」，
与本仓库的实际预处理不符）。rev_probe1 里 `@张三 …` 一类「HEAD=False→新=True」的差异，
在生产上因为前置剥离而不存在，不构成风险。

### 【note-13】写路径 `/api/bot/pull-requests/batch` 未接入共享选择器

`keytao-next/app/api/bot/pull-requests/batch/route.ts:287` 在无 `batchId` 时直接
`randomUUID()` 建新批次，不走 `findCurrentBotDraftBatch`。目前不会分叉（bot 侧总是先解析
batch_id，只有真的没草稿时才传 null），且事务内有 `assertNoOtherBotDraftWithContent`
兜底，但与 `batch-draft` 路由的写法不一致。建议统一，或加注释说明它「从不隐式解析」。

### 【note-14】`_format_draft_response` 每次渲染多一次未锚定读

`keytao-bot/keytao_bot/plugins/openai_chat.py:5932-5948`：即使已有 anchor，也要先发一次
不带 batch_id 的 `keytao_list_draft_items` 来探测指针漂移，命中漂移才重读。
B1 之后该读不再建批次，安全；只是每次草稿渲染多一个 RTT。可接受。

### 【note-15】B4 存量清理方案复核通过

`已归档空草稿（原：键道助手…）` 前缀确实能让批次从所有 `startsWith('键道助手')` 指针查询里消失；
Step 2 的 `LIKE '已归档空草稿（原：键道助手%'` 与 Step 1 一致；
`contentVersion = 0` + 无 PR + 1 天时间窗 + 通告锁的组合合理；外键分析（`pull_requests.batchId`
ON DELETE CASCADE、其余为 batch 指出的可空外键）正确。仅设计未执行，符合禁区。

---

## 按维度的收口

- **R1 安全语义**：红线核查通过（`_MUTATION_INTENT_RE` 等五个定义与 HEAD 全等）；
  suggestedCommand 自检机制真实有效（新一轮回放全通过）；A6 第二次调用完整重跑绑定 + CAS，
  没有绕过用户授权的新路径；只读轮的结构化回包**不含**票据/digest，不泄露可执行确认路径。
  **但**发现 major-1（授权层被实质扩权，实现者的免责论断被实测证伪）与 major-5、minor-6。
- **R2 正确性与事故闭环**：事故 7/8/9 三句（「执行顺延…」）已能一次授权直达执行；
  撤回后「吃席」在 B1+B2+A7 三重作用下可见（内存假 prisma 端到端回归已证）。
  **但**第 1/6 句落到只读轮，仍无确定性补救路径（major-4）；
  跨仓协议适配漏了 word_discovery 的真实错误契约（major-2）；部署顺序被强耦合（major-3）；
  A7 锚定与 B2 选择器组合无缝隙（锚定读走 `?batchId=`，服务端仍按 `creatorId` 收敛，
  批次非当前草稿时 items 路由返回空快照并触发显式漂移告警）。
- **R3 测试质量**：4 次变异全部检出，测试是真语义测试；被改写的 `security-guards.test.ts`
  新断言完整覆盖新语义（`batch.create` 未被调用 + 响应 `batchId:null/exists:false`）
  且保留了 `creatorId: 2` 的权限绑定断言。**但**见 minor-10（A3 覆盖过薄）与 major-2
  （word_discovery 新分支零覆盖，且现有断言与新实现互斥）。
- **R4 工程质量**：无越界改动（`_stage_agent_mutation` 死代码未删、keytao-next 未被 A 工作流触碰、
  未 commit/push/连库）；无 TODO/print/console.log 残留；lint + tsc 干净；风格与仓库一致。
  **但**见 minor-9（不可达守卫）与 note-12（注释与实际预处理不符）。

---

# 修复轮复核（round 2）

复核方式与上轮一致：只读代码审查 + `keytao-bot/.venv` 内对真实函数/真实 HTTP 层的对抗性探针
+ 变异检验。**两个仓库在本轮复核前后逐字节一致**（`git diff` / `git status --short` 全等；
bot HEAD 仍 `e5dfd27`、next 仍 `1c2b4e3`，均未 commit）。

本轮探针：`rev2_probe1_authz.py`（major-1 + 等价性 + 45 句回归扫描）、
`rev2_probe2_authz_tool.py`（新工具对抗性核查）、`rev2_probe3_operand_and_batch.py`
（候选补丁验证 + minor-6 守卫）、`rev2_probe4_batch_launder.py`（minor-6 绕过 PoC）、
`rev2_probe5_cas_contract.py`（bot 请求体 × next 路由分支）、
`rev2_probe6_word_discovery.py`（真实 `keytao_json` + 真实状态码的五步链路）。

## 逐条 pass / fail

### 1.【major-1】授权层扩权 — **PASS**（残留一例已验证等价）

- 红线：`_EXPLICIT_REQUEST_PREFIX_RE` 与 HEAD **字符串完全相同**（撤回成功）；
  `_MUTATION_INTENT_RE` 仍与 HEAD 全等；`_COMMAND_PREFIX_RE` 保留「执行」；
  `message_authorizes_mutation` 末尾改为先 `_COMMAND_PREFIX_RE.sub` 再判 0 位
  （`tools.py:512-528`）——正是上轮实测验证过的那条修法。
- 上轮 8 句 hazard：**7 句回到 False**，只剩「执行保留策略」。
- **等价性论断实测成立**：`NEW(执行保留策略)=True`、`NEW(保留策略)=True`、`HEAD(保留策略)=True`，
  8 组「整句 / 剥前缀后」判定三方全等，无一例外。即它不是本轮引入的新授权类。
- 45 句闲聊/引用/否定/取消回归扫描：**newly raised vs HEAD = 0**；
  25 句必须授权（含 A2 全部验收句式）**25/25 仍为 True**。
- 残留（**非本轮引入，P2**）：`确认结果保留一下 / 请结果保留一下 / 现在结果保留一下 /
  直接日志保留三天` 等「可剥前缀 + 保留」句在 HEAD 上就已是 True。根因是
  `保留` 同时是 `_MUTATION_INTENT_RE` 成员与保护词。要根治须把「保留」移出意图表（禁区），
  建议单独立项。

### 2.【major-4/5】只读轮授权换取工具 — **PARTIAL（③ pass，①部分，② / ④ fail）**

- **③ 动作类别不匹配拦截：PASS，7/7 全对。**
  用户说加词→模型要删除/提交/撤回/顺延，一律不给指令；类别一致才给。
- **① 无改动意图不给指令：基本 PASS，一处缺口。**
  提问（「吃席到底怎么打 wkxk」「现在草稿里有什么？」）、解释、否定、取消、被标记为引用的引号内容、
  空消息 —— **全部不给**，且工具本身不挂进清单（`message_mentions_change_request=False`）。
  缺口：**转述句「他说顺延吃席到 wkxk」仍会给出 `@我 顺延「吃席」到 wkxk`**——
  `message_requests_change` 不做授权层那套「动词须在 0 位 / 转述过滤」。
- **② 参数滥用：FAIL。** `message_requests_change` 只校验**动词类别**，
  从不校验 `word` / `code` / `pr_id` 是否出现在用户原话里。实测：用户发事故原句
  「把吃席的编码放在赤溪前面」，模型把参数换成任意值即可拿到
  **`@我 顺延「攻击者选的词」到 zzzz`**（`rev2_probe2` §2）。
- **④ 门槛过宽：FAIL。** `_POSITIONAL_CHANGE_RE`（`tools.py:534-538`）含
  `放在|放到|调到|挪到|排在|插到|插入|抢占|占用|提前|前面|后面|往前|往后|位置`。
  实测 10/10 纯闲聊全部通过门槛并拿到现成指令，其中
  「提前… / 占用率有点高 / 放到明天再说 / 调到静音模式 / 插入一张图片看看 / 排在我后面的是谁」
  **正是核验报告 round-0 点名的日常闲聊清单**：

  ```
  requests_change=True  exposed=True  sugg=@我 顺延「吃席」到 wkxk   我前面说错了
  requests_change=True  exposed=True  sugg=@我 顺延「吃席」到 wkxk   占用率有点高
  requests_change=True  exposed=True  sugg=@我 顺延「吃席」到 wkxk   放到明天再说
  requests_change=True  exposed=True  sugg=@我 顺延「吃席」到 wkxk   调到静音模式
  requests_change=True  exposed=True  sugg=@我 顺延「吃席」到 wkxk   插入一张图片看看
  ```

- **合并影响**：②+④ 意味着 —— **只要用户这轮说了任何一个位置类词，模型就能拿到一条
  校验器保证通过、参数完全由模型指定的授权指令**，并被文案要求「原样转述给用户」。
  这不授予任何写权限（`writes_allowed` 仍为 False，工具仍被摘除，不执行任何写入），
  但它把 major-5 的社工面从「任何一轮」只收窄到「任何含位置词的一轮」，收窄幅度有限。
  事故中的用户群体正是会逐条照做的。**建议部署前补掉**（改动很小）。
- **已实测的修法**：在 `self_checked_suggested_command` 里加一道**操作数在场**校验——
  词类工具要求 `_contains_exact_target(context.current_message, word)`，
  id 类工具要求 id 出现在原文；`submit` / `recall` 无操作数照旧放行。
  实测矩阵（`rev2_probe3` §A）：合法用例 **5/5 全保**（含事故第 1 句、
  「把吃席顺延一下」、「删掉草稿条目 12」、「提交草稿吧」），
  滥用/闲聊用例 **7/7 全挡**。另建议把 `占用|提前|前面|后面|位置` 从
  `_POSITIONAL_CHANGE_RE` 移除或要求与编码 token 共现。

### 3.【major-2】`allow_status` + word_discovery — **PASS**

- `allow_status` 是**关键字专用**参数（`http_client.py:381-386`），默认 `None`；
  全仓 grep 只有 **1 个生产调用点**（`word_discovery.py:1690`，`{400}`），没有别的调用点被顺手放宽。
- 放行范围正确（实测）：`allow_status=None → 500 抛`、`{400} → 500 仍抛`、`{500} → 返回体`。
  只对显式列出的状态码放行。
- **真实契约端到端实测**（探针挂在 `keytao_request` 之下，跑的是真的 `keytao_json`）：

  ```
  GET  /api/bot/batches/latest-draft                       (batchId=null)
  POST /api/bot/pull-requests/batch-draft  confirmed=False        → 400 + requiresConfirmation
  POST /api/bot/pull-requests/batch-draft  confirmed=True  batchId=temp-uuid-1 ver=0
  POST /api/bot/batches/created-1/submit
  POST /api/bot/batches/created-1/auto-approve
  success=True   confirmed batch-draft writes reaching the server: 1   （上轮实测为 0）
  ```

  409（`assertNoBotDraftBatch` 触发）路径也验过：干净降级、`pendingRecovery` 为空。
- 测试确实覆盖：链路断言已改为 5 步，并断言 `allow_status` 含 400、
  确认请求回带服务端 `contentVersion` + `warningDigest`，另加 8 条降级用例。

### 4.【CAS 契约双侧一致性】— **PASS（无 blocker），2 处文档层偏差**

逐字段核对结果：

| 契约字段 | B 侧（next） | A 侧（bot） | 结论 |
|---|---|---|---|
| 无草稿读返回 | `latest-draft`：`batchId:null, exists:false, contentVersion:0` (`route.ts:47-58`)；`items`：`batchId:null, contentVersion:0, items:[]` | shift 取基线 `("", 0)`（`skills tools.py:3816-3823`） | **一致** |
| 预览 400 体 | 两个写接口都在 `confirmed!==true` 时返回 400 + `requiresConfirmation/batchId/contentVersion/warningDigest` | `_keytao_strict_batch_add_to_draft` 见 `requiresConfirmation` 直接回传；word_discovery 用 `allow_status={400}` 读体 | **一致** |
| 确认请求体（省略 batchId） | `batchId` 缺失 + `expectedContentVersion:0` → 允许（`batch-draft/route.ts:94`），走 `batchIsVirtual` | strict-add **整键省略**（`skills tools.py:3572-3574`） | **一致** |
| 确认请求体（回传临时 uuid） | `findUnique` 未命中 + `confirmed && version===0` → `batchIsVirtual`（`batch-draft/route.ts:150-158`）；`/batch` 同理 | word_discovery 回传 `temp-uuid` | **一致** |
| 常量哈希身份 | `NEW_BOT_DRAFT_BATCH_IDENTITY='new-bot-draft-batch'`，两个写接口在批次不存在时都用它当 `warningState.targetBatchId` | bot 不算 warningDigest，N/A | **一致**（preview/confirm digest 可复现） |
| 409 处理 | `BatchContentLockedError.status=409`，两个路由的 catch 都按 `error.status` 返回 | HTTP 409 → `staleConfirmation:True`（`skills tools.py:3617-3623`），orchestrator 见 `staleConfirmation` 保留预览不重试 | **一致** |

- **偏差 A（minor，文档 vs 实现）**：契约 §6.3 写「期间只要冒出任何草稿（**含空草稿**）→ 409」。
  实测（`rev2_probe5`）：**省略 batchId** 的确认遇到一个 `contentVersion=0` 的空草稿时
  **不报 409，而是直接写进那个空草稿**——因为 `findCurrentBotDraftBatch` 命中后
  `existingBatchSnapshot` 为真、版本又恰好相等，`assertNoBotDraftBatch` 根本不会被调用。
  **回传临时 uuid** 的确认则走 `batchIsVirtual` → `assertNoBotDraftBatch` → 真的 409。
  即契约声称可互换的两种确认形态，CAS 严格度不同。
  实际危害为零（写进一个版本 0 的空草稿与新建一个批次等价，且顺延还有 bot 侧
  `batch_id != current_batch_id` 的 stale 检查兜在前面），但**契约文本需要改**，
  否则后续实现者会按错误的强度假设写代码。
- **偏差 B（note，文档错标接口）**：契约 §6.3 标题写「顺延走 `POST /api/bot/pull-requests/batch-draft`」，
  实际顺延写入走 `_keytao_strict_batch_add_to_draft` → **`POST /api/bot/pull-requests/batch`**
  （`skills tools.py:3566`）。两个接口都实现了同一套 CAS 语义，所以没有功能问题，但文档指错了路。
- **测试缺口（minor）**：`test_shift_phrase_code_works_with_no_draft_batch` mock 的是
  `_keytao_strict_batch_add_to_draft`，**停在 HTTP 层之上**；两侧握手（真实请求体 × 真实路由分支）
  目前没有任何测试覆盖，只有本次评审的探针覆盖过。建议补一条按真实请求体断言的契约测试。

### 5.【终验】— **PASS，数字逐个复现**

| 套件 | 实现者报告 | 复核实测 |
|---|---|---|
| `test_memory_safety.py` | 125 | **125** ✓ |
| `test_state_machine.py` | 1195 | **1195** ✓ |
| `test_word_discovery.py` | 261 | **261** ✓ |
| `test_security_fixes.py` | 207 | **207** ✓ |
| `test_review_gate.py` | 150 | **150** ✓ |
| `test_llm_policy.py` | 9 | **9** ✓ |
| `test_isolation_fixes.py` | 67 | **67** ✓ |
| `test_skill_hardening.py` | 67 | **67** ✓ |
| `test_image_input.py` | 30 | **30** ✓ |
| keytao-next `vitest --config vitest.unit.config.ts` | 51 文件 / 521 例 | **51 / 521** ✓ |
| `./node_modules/.bin/tsc --noEmit` | 0 错误 | **exit 0** ✓ |

未运行 `pnpm test`（禁令遵守）。

### 6.【上轮 minor/note 抽查】

| 编号 | 结论 | 证据 |
|---|---|---|
| **⑥** untrusted_batch_reference | **PARTIAL —— 守卫本身生效，但可两跳绕过** | 见下 |
| **⑦** 去重键 | **PASS** | 键为 `(blockReason, tool_name, 参数指纹)`（`orchestrator.py:1249-1277`），调用点确实传入 `fn_name` + `canonical_fn_args`（`:587-593`） |
| **⑨** 不可达守卫 | **PASS** | 守卫已删；改为「调用方已给出 `expected_content_version` 就不再重新解析批次」，这条有实际作用（防止悄悄采纳计划后冒出的草稿）；请求体在无批次时整键省略 `batchId` |
| **⑩** A3 覆盖过薄 | **PASS** | 变异复验：撤销空白保留 → **3 条红**（`test_authorization_view_keeps_token_boundaries` / `test_separate_ids_do_not_merge_into_one_token` / `test_every_suggested_command_passes_its_own_validator`）；邻接过滤回退 → **2 条红**（`test_code_written_next_to_the_word_still_binds` / `test_incident_shift_phrasings_all_authorize_and_bind`）。两半各自有敏感用例，头牌测试不再迟钝 |
| **⑪** 保护词被整句吞掉 | **PASS** | 实测：`把「保留」顺延到 wkxk` 执行；`把「保留」顺延到 wkxk，保留原编码` / `顺延保留到 wkxk，保留原来的编码` / `请把吃席顺延到 wkxk，但保留赤溪` **全部拦截**。只忽略被引号点名的那一处 |
| **⑫** 注释与预处理不符 | **PASS** | 注释已改为「本仓库插件已剥离 @，该正则存在的意义是让 `@我 …` 建议按同一套校验器自检」 |
| **⑭** 多余未锚定读 | **PASS** | 未锚定读只剩预览本身（兼指针探针），指针一致时其结果即所需；条目读取有锚点时始终锚定（`openai_chat.py:_format_draft_response`） |

#### ⑥ 的两跳绕过（PoC 已跑通）

`BATCH_ANCHORED_READ_TOOLS` 只覆盖两个读工具，而 `batch_id` 在**写工具**上仍是未受检的
未声明参数；`_inject_known_batch_url` 用 `data.setdefault("batchId", batch_id)`
把模型给的 id 原样盖进结果（含**错误**结果），`_collect_trusted_batch_ids` 再把它收进可信集：

```
step 0  keytao_get_batch_preview(batch_id=victim)        -> untrusted_batch_reference   （守卫生效）
step 1  keytao_create_phrase(word/code 均合法绑定, batch_id=victim)
          tool kwargs delivered: {... 'batch_id': 'victim-batch-uuid-0000' ...}
          result:  {'success': False, 'message': '无权限操作此批次', 'batchId': 'victim-batch-uuid-0000'}
          trusted_batch_ids now: {'victim-batch-uuid-0000'}
step 2  keytao_get_batch_preview(batch_id=victim)        -> ALLOWED（打到无鉴权的公开 preview 接口）
```

- 失败场景：写轮（用户给了一条合法写指令）里，注入内容诱导模型顺手带一个别人的 batchId，
  两跳之后把别人的草稿 diff 打印到群里。该 preview 接口本来就是公开的（不构成平台新漏洞），
  但这正是 ⑥ 想堵的 LLM 可控读取。
- 建议修法（二选一或都做）：① `_collect_trusted_batch_ids` 跳过「模型本次调用里自己传进来的
  同一个 batch_id」，或只从 `success`/`requiresConfirmation` 的服务端结果里收；
  ② `_inject_known_batch_url` 对失败结果改用 `requestedBatchId` 而不是 `batchId`；
  ③ 把可信批次校验从两个读工具扩到**所有**接受 `batch_id` 的工具。

## 剩余风险与 P2 遗留清单

**建议部署前修（各自改动很小，且修法已实测）**

1. **建议指令缺少操作数校验**（major-5 残留，item 2②④）——
   `tools.py:self_checked_suggested_command` 加操作数在场校验；
   收紧 `_POSITIONAL_CHANGE_RE`。验证矩阵：合法 5/5 保、滥用 7/7 挡。
2. **可信批次集可被两跳污染**（⑥）—— `_collect_trusted_batch_ids` / `_inject_known_batch_url` 二选一改。

**可作为 P2 / 下一批**

3. 转述句（「他说顺延…」）仍能换到现成指令 —— 给 `message_requests_change` 补一层转述/引述过滤。
4. `保留` 同时是意图动词与保护词，导致「确认/请/现在 + …保留…」类句授权（**HEAD 既有**，非本次引入）。
5. CAS 契约文本两处需订正：省略 batchId 与回传临时 uuid 的 409 严格度不同；顺延实际走 `/pull-requests/batch`。
6. 两侧握手（真实请求体 × 真实路由分支）无测试覆盖，建议补契约测试。
7. A6「一次授权即执行」在 HTTP 层仍会先收到一次 warningDigest 挑战（`/batch` 对
   `confirmed!==true` 无条件返回 `requiresConfirmation`，而 auto-confirm 不带
   `expected_warning_digest`），即「一条用户指令 + 一张风险票据」。这是实现者自述的降级路径
   （从两张票降为一张票），非回归，但现有测试都停在 `_keytao_strict_batch_add_to_draft` 之上，
   未验证过这一层。
8. 上轮 note-13 已由 B 侧实现（`/pull-requests/batch` 接入共享选择器），本轮复核代码确认。
9. 上轮报告的 P2 项（引用确认 digest、`[不可信参考资料]` 拆独立消息、`build_reply_context`
   无条件注入、`confirmed` schema 自相矛盾）仍未处理。

## 部署结论

**事故闭环与安全底线已达到可部署状态**，前提是遵守既定顺序：**先发 keytao-bot，再发 keytao-next**
（已复核该顺序安全：旧 next 的 get-or-create 永远不返回 `batchId:null`，新 bot 的 None 透传、
CAS-on-absence、preview→confirm 三条新路径在旧服务端上都不触发；反向顺序会让旧 bot 在
「草稿全部提审完」这一常见状态下无法加词）。

无 blocker。上表第 1、2 两项建议在同一批里补掉——它们不阻断执行、不授予写权限，
但第 1 项是一条从注入到「用户被指令去授权」的可用社工路径，改动小且修法已验证。
