# Workstream A 改动摘要（keytao-bot 授权/状态机修复）

仓库：`/Users/rea/code/keytao-org/keytao-bot`（工作区改动，未 commit / 未 push）
实现者：Opus 5 兜底通道 | 日期：2026-08-04
**第 2 轮**：已按独立评审报告（`keytao-fix-review.md`）修复 major-1/2/4/5、minor-6/7/9/10/11、
note-12/14，并按 keytao-next 新定义的 CAS-on-absence 契约解锁修复 minor-8。逐条见 §5。
**第 3 轮**：按复核轮（`keytao-fix-review.md` §修复轮复核 round 2）补掉两处残留——
建议指令的**操作数在场校验**（major-4/5 残留）与**可信批次集两跳污染**（minor-6 残留）。见 §7。

---

## 1. 文件 → 改动点 → 规格编号

### `keytao_bot/harness/tools.py`

| 改动点 | 规格 |
|---|---|
| `ToolContext` 增加 `attachment_context` 字段，只有真的带附件/视觉通路才为 True | A1 |
| 新增 `BLOCK_REASON_*` 枚举（source_untrusted / verb_not_matched / binding_incomplete / ticket_required / bulk_delete_not_requested / manual_shift_forbidden / batch_too_large）与 `policy_block()` 构造器；`_validate_policy` 与 `_validate_current_message_binding` 的 13 处拦截全部改为结构化返回（`blockReason` + `missing` + 可选 `suggestedCommand`） | A1 |
| 删除写死文案「历史、记忆、引用或附件内容不能授权修改草稿或提交」。纯文本路径改为「这条消息里没有识别到明确的执行指令（与历史、记忆或引用无关）」；附件路径保留准确的那半句「附件内容不能授权…」 | A1 |
| 新增 `_suggested_command_text()` / `_suggested_item_command()` / `self_checked_suggested_command()`：按工具+参数生成补救指令，**先用同一套 `message_authorizes_mutation` + `_validate_current_message_binding` 自检**，通过才输出；一律带 `@我 ` 前缀（`SUGGESTION_MENTION_PREFIX`）。自检不过则不给建议，只说明原因 | A1 |
| 新增 `_LEADING_MENTION_RE`，在 `_mutation_authorization_view` 开头剥掉一个前导 `@xxx ` 提及（保证「@我 …」建议在把 @ 保留进正文的平台上同样可用） | A1 |
| `_COMMAND_PREFIX_RE` 加入「执行」；`message_authorizes_mutation` 末尾改为「先用 `_COMMAND_PREFIX_RE` 剥前缀再判动词是否在 0 位」（第 2 轮修正，见 §5 major-1） | A2 |
| `_WORD_LEFT_PREFIXES` 加「顺延/移到/挪到/改到」；`_WORD_RIGHT_SUFFIXES` 加「顺延/的编码/的代码/到/移到/挪到/改到」 | A2 |
| `_mutation_authorization_view`：判定仍用去空格串（前缀/动词继续在 0 位匹配），但**存入视图的是空白折叠为单空格的版本**，保留 token 边界 | A3 |
| `_explicit_code_spans`：`distance == 0` 过滤改为「仅排除与目标 span 真正重叠的 token」，紧邻目标的编码不再被丢弃 | A3 |
| `_action_is_bound_to_target`：与目标重叠的动作 token 直接跳过（词条本身是「删除/保留」时不能给自己当动词） | A4 |
| `trusted_mutation_source`：引号内容命中动词表**不再**整段抹除；新增 `_quoted_span_is_command()`——只有引号框架被标为不可信（引用/复述/备注…），或引号内容同时含动词与操作对象（草稿/批次/条目/全部/所有/编码类 token，或长度 > 8）时才 redact | A4 |
| 新增 `_has_protection_outside_target()` + `_quoted_target_spans()`；顺延分支的「全句含保护词就拒绝」改为「只忽略用户用引号点名为词条的那一处」——句中其它位置的「保留原编码」仍然拦截 | A4 / minor-11 |
| 新增 `message_requests_change()` / `message_mentions_change_request()`：**只用于决定要不要给建议指令，不授予任何权限**。它比授权层宽（接受「放在/调到/排在」等禁区词），也比授权层严（问句、解释、否定、取消一律不算）。`self_checked_suggested_command` 以它为门槛，取代原来的 `writes_allowed` | major-4 / major-5 |
| 第 3 轮新增 `_suggestion_operands()` / `_operands_are_present()`：建议里要提到的每一个实体（word / old_word / pr_id / ids）**必须已经出现在用户本轮原话里**，否则不给建议；submit / recall 无操作数照旧。同时从 `_POSITIONAL_CHANGE_RE` 移除纯位置词「占用\|提前\|前面\|后面\|位置」 | major-4/5 残留 |
| 第 3 轮：可信批次校验从两个读工具扩到**所有**带 `batch_id` 的工具调用（写工具的未声明 `batch_id` 同样被拦），杜绝「先用写工具把陌生批次号洗进结果」的两跳绕过 | minor-6 残留 |
| 新增 `BLOCK_REASON_UNTRUSTED_BATCH` + `BATCH_ANCHORED_READ_TOOLS`：模型发起的读取只能锚定本轮服务端返回过的批次；内部调用（无 `current_message`）不受限 | minor-6 |

### `keytao_bot/harness/orchestrator.py`

| 改动点 | 规格 |
|---|---|
| 工具清单在 `visual_context` 之外增加按 `mutations_allowed` 过滤：只读轮移除全部 `MUTATING_TOOL_NAMES`，并追加一条 system 说明「本轮只读，请原样转述 suggestedCommand」 | A5 |
| 模型仍点名被摘掉的写工具时，不再抛 `unknown tool`（会退化成「工具参数格式错误」），而是回一条结构化拦截（含自检过的 `suggestedCommand`），让模型能向用户解释 | A5 |
| `ToolContext` 传入 `attachment_context=bool(visual_context)` | A1 |
| 新增 `_auto_confirm_shift_plan()`：顺延返回 `requiresConfirmation/shiftPlan` 且本轮 `writes_allowed` 时，直接用服务端返回的 `planDigest`+`batchId`+`contentVersion` 立即回调一次；第二次调用**完整重跑策略校验**，服务端 CAS 不变。执行后 `canonical_fn_args` 替换为实际执行的参数 | A6 |
| 新增 `_server_plan_binding()`；`_save_pending_tool_confirm` 为顺延票据合并服务端 digest/批次/版本（消灭「票据 A 是废票、必须再换一张票」这一态） | A6 |
| 新增 `_deduplicate_block_reason()` + `reported_block_reasons`：去重键为 **(blockReason, 工具名, 参数指纹)**，同一条被拒调用第二次起只回一句短提示（`repeatedBlock: true`，去掉长补救文案与 suggestedCommand）；同一轮里另一个工具/另一个目标仍能拿到完整说明 | A8 / minor-7 |
| 新增只读轮专用工具 `keytao_request_write_authorization`（`WRITE_AUTHORIZATION_TOOL`）：不写任何数据，把模型拟调用的写工具+参数换成一条自检过的 `suggestedCommand`。仅当**本轮消息确实在要求某种改动**时才挂进工具清单；纯提问轮不挂 | major-4 |
| 新增 `_collect_trusted_batch_ids()`，把服务端返回过的 batchId 收进 `ToolContext.trusted_batch_ids`；第 3 轮增加「**调用方自己传进来的 batch_id 一律不因结果回显而被信任**」，并把收集时机移到 shift 自确认之前（自确认用的是服务端预览给出的批次，仍然合法） | minor-6 |

### `keytao_bot/harness/state.py`

| 改动点 | 规格 |
|---|---|
| 新增 `invalidate_actor_related(actor_key, batch_id=...)`：只作废「草稿域工具 + 锚定同一批次（或无锚点）」的票据；`PendingAddWord` 等无关待确认一律保留。`delete_actor` 保留未动 | A6 |

### `keytao_bot/plugins/openai_chat.py`

| 改动点 | 规格 |
|---|---|
| `_format_draft_response()` 增加 `batch_id` 参数，并默认回退到 `data["batchId"]`；预览/列表读取按该批次锚定 | A7 |
| 指针漂移检测：唯一一次未锚定读取就是**预览本身**（它同时充当指针探针），指针与锚点一致时其结果即所需，无任何被丢弃的读取；不一致才重取预览，并在回复顶部输出「⚠️ 当前草稿指针与上次操作的批次不一致：指针指向 X，本次批次是 Y」。条目读取在有锚点时**始终**锚定 | A7 / note-14 |
| `_execute_confirmed_tool` 的顺延票据守卫：空 batch_id 仅在 `expected_content_version == 0` 时视为合法基线 | minor-8 |
| 撤回路径传 `batch_id=exact_batch_id` | A7 |
| `_try_handle_draft_recall_command` 的 `delete_actor` → `invalidate_actor_related(..., batch_id=撤回批次)` | A6 |

### `keytao_bot/skills/keytao-draft/tools.py`

| 改动点 | 规格 |
|---|---|
| `_fetch_draft_snapshot()` 增加 `batch_id`；create / remove / batch-add / strict-batch 四处调用改为传入本次写入的批次 | A7 |
| `keytao_get_batch_preview()` 增加 `batch_id` 参数（给定时不再走 latest-draft 指针） | A7 |
| `get_latest_draft_batch()`：明确 `batchId:null` = 「暂无草稿」而非失败（404→UserNotFoundError 逻辑保留不动） | keytao-next 配套 ①② |
| 三处 `batch_id=None` 早退「无法获取草稿批次」删除（create_phrase / batch_add_to_draft / _keytao_strict_batch_add_to_draft），改为把 None 透传给写接口由服务端按需创建；仅在「带确认票据却没有目标批次」时才拒绝 | keytao-next 配套 ① |
| `keytao_shift_phrase_code`：无草稿时基线取 `("", 0)` 并照常进 `planDigest`；确认写入传 `batch_id=None` + `expected_content_version=0`，由服务端在「当时确实没有草稿」的断言下建批次 | minor-8 |
| `_keytao_strict_batch_add_to_draft`：调用方已给出 `expected_content_version` 时不再重新解析批次（否则会悄悄采纳计划之后冒出来的草稿，架空 CAS）；请求体在无批次时**整个省略 batchId**。同时删掉 minor-9 指出的不可达守卫 | minor-8 / minor-9 |
| `keytao_get_batch_preview` / `keytao_list_draft_items` 的 `batch_id` 正式写进 schema（不再是隐藏参数），并注明必须是本轮出现过的 batchId | minor-6 |

### `keytao_bot/utils/word_discovery.py`

| 改动点 | 规格 |
|---|---|
| `submit_discovered_words` 的 batch-draft 写入改为「先问后确认」：首次 `confirmed:false` 且 `allow_status={400}`（服务端正是用 **HTTP 400 + requiresConfirmation** 携带确认快照的），再用返回的 `contentVersion` + `warningDigest` 发 `confirmed:true`。原来无条件 `confirmed:true` 且不带 digest → 服务端必回 400 → 从未写入成功 | keytao-next 配套 ③ / A6 |
| （配套）`http_client.keytao_json` 增加 `allow_status`：只有显式列出的非 2xx 才把响应体交回调用方判断，其余仍照旧抛 `KeytaoApiError` | keytao-next 配套 ③ |
| latest-draft 无草稿（batchId 为空）不再判失败，写入后从响应里取批次 id 供 submit/auto-approve 使用；**非法** batch id 仍然拒绝 | keytao-next 配套 ①③ |

### 测试

| 文件 | 新增 | 规格 |
|---|---|---|
| `test_memory_safety.py` | `ShiftAuthorizationTests`（6）：事故 12 种顺延句式全通过、拦截原因诚实 + 建议指令可原样重放、7 个工具的 suggestedCommand 逐个自检、引号词条正例（保留/提交/修改）、引号整条命令仍被 redact、带引用前缀的引号仍不可信 | A2/A3/A4/A9 |
| `test_memory_safety.py` | `ReadOnlyTurnToolExposureTests`（4）：只读轮不暴露写工具、写轮仍暴露、被摘工具给结构化原因而非协议错误、同一原因一轮只详细说明一次 | A5/A8/A9 |
| `test_memory_safety.py` | `ShiftSingleAuthorizationTests`（2）：一次授权直接执行且不留票据、退化到票据时票据带服务端计划 | A6/A9 |
| `test_state_machine.py` | `test_recall_shows_items_from_the_recalled_batch`（6 checks）：mock 服务端指针漂移，断言撤回后能看到被撤回批次的条目 + 明确告警 + 读取被锚定 | A7/A9 |
| `test_state_machine.py` | `test_recall_only_voids_tickets_bound_to_the_recalled_batch`（3 checks） | A6/A9 |
| `test_memory_safety.py` | 第 2 轮：「执行+保护词」8 条闲聊负例、「剥前缀不得引入新授权类」等价性断言 | major-1 |
| `test_memory_safety.py` | 第 2 轮：`test_a_question_never_receives_a_ready_made_authorization`（5 例：提问/看图/动作不匹配都拿不到现成指令） | major-5 |
| `test_memory_safety.py` | 第 2 轮：`ReadOnlyAuthorizationRequestTests`（4）：有改动意图的只读轮才挂授权换取工具、提问轮不挂、换回的指令可原样执行、提问轮硬要调也只给原因 | major-4 |
| `test_memory_safety.py` | 第 2 轮：token 边界与紧邻编码各自的敏感用例（`test_authorization_view_keeps_token_boundaries` / `test_separate_ids_do_not_merge_into_one_token` / `test_code_written_next_to_the_word_still_binds`） | minor-10 |
| `test_memory_safety.py` | 第 2 轮：无草稿顺延一次授权即执行 + 「空批次仅在版本 0 时算基线」负例 | minor-8 |
| `test_state_machine.py` | 第 2 轮：`test_shift_phrase_code_works_with_no_draft_batch`（9 checks，按 CAS-on-absence 契约 mock 服务端） | minor-8 |
| `test_word_discovery.py` | 第 2 轮：链路断言改为 5 步真实 preview→confirm（含 `allow_status={400}`、确认必须回带服务端快照），新增「400 直接拒绝」「确认快照不可用」两条降级用例 | major-2 |
| `test_memory_safety.py` | 第 3 轮：`test_a_suggestion_can_only_name_what_the_user_named`（5 例，评审 PoC：事故原句 + 模型自选参数）、`test_position_words_alone_never_produce_a_command`（10 句闲聊） | major-4/5 残留 |
| `test_memory_safety.py` | 第 3 轮：`TrustedBatchAnchorTests`（3）：两跳洗白 PoC 逐跳被拦、服务端返回的批次仍可用、内部调用不受限 | minor-6 残留 |

---

## 2. 验收数据

### 测试套件（全绿）

| 套件 | 修复前 | 第 1 轮 | 第 2 轮 | 第 3 轮（当前） |
|---|---|---|---|---|
| `test_memory_safety.py` | 100/100 | 113/113 | 125/125 | **130/130**（+30 新增） |
| `test_state_machine.py` | 1177/1177 | 1186/1186 | 1195/1195 | **1195/1195**（+18 checks） |
| `test_security_fixes.py` | 207/207 | 207/207 | 207/207 | **207/207** |
| `test_review_gate.py` | 150/150 | 150/150 | 150/150 | **150/150** |
| `test_llm_policy.py` | 9/9 | 9/9 | 9/9 | **9/9** |
| `test_isolation_fixes.py`（附加保护集） | 67/67 | 67/67 | 67/67 | **67/67** |
| `test_word_discovery.py`（附加） | 256/256 | 256/256 | 261/261 | **261/261**（真实契约） |
| `test_skill_hardening.py`（附加） | 67/67 | 67/67 | 67/67 | **67/67** |
| `test_image_input.py`（附加） | 30/30 | 30/30 | 30/30 | **30/30** |

规格点名的保护集（`test_word_substring_does_not_bind_a_different_word` /
`test_code_requires_exact_token_or_same_turn_capability` /
`test_batch_codes_cannot_be_swapped_between_targets` /
`test_fully_bound_agent_mutation_reaches_write_sink_without_local_ticket`）全部保持绿。

### 21 行矩阵：5/21 → **16/21**

目标 `word=吃席`、`target_code=wkxk`、`trusted_codes={wkxk,wkxkv,wkxkoo}`。

| 句式 | 修复前 | 修复后 | 说明 |
|---|---|---|---|
| 顺延 吃席 wkxk | ❌ | ✅ | A3 空格保留 + 紧邻编码不再丢 |
| 顺延：吃席 wkxk | ❌ | ✅ | A3 |
| 顺延吃席到wkxk | ❌ | ✅ | A2 左前缀「顺延」+ 右后缀「到」 |
| 把吃席顺延到 wkxk | ❌ | ✅ | A2 右后缀「顺延」 |
| 请把吃席顺延到 wkxk | ❌ | ✅ | A2 |
| 把「吃席」顺延到 wkxk | ✅ | ✅ | — |
| 顺延「吃席」到 wkxk | ✅ | ✅ | — |
| 确认顺延：吃席 → wkxk | ✅ | ✅ | — |
| 确认顺延：吃席 → wkxk，赤溪顺延 | ✅ | ✅ | 事故原句 |
| 确认顺延: 吃席 wkxk | ❌ | ✅ | A3 |
| 执行顺延：吃席 → wkxk | ❌ | ✅ | A2「执行」前缀 |
| 确认执行顺延：吃席 → wkxk | ❌ | ✅ | A2 |
| 顺延，吃席，wkxk | ❌ | ❌ | 逗号把词/码切成独立子句，子句无动词被丢弃 |
| 修改吃席 wkxk | ❌ | ✅ | A3 |
| 把吃席改成 wkxk | ✅ | ✅ | — |
| 把吃席的编码改成 wkxk | ❌ | ✅ | A2 右后缀「的编码」 |
| 把吃席的编码改为 wkxk | ❌ | ✅ | A2 |
| 吃席顺延 wkxk | ❌ | ❌ | 动词不在 0 位且无显式请求标记；扩权需动 `_MUTATION_INTENT_RE`（禁区） |
| 吃席 改成 wkxk | ❌ | ❌ | 同上 |
| 把吃席的编码放在赤溪前面 | ❌ | ❌ | 「放在」属禁区词，未入表；**生产上（只读轮）通过 `keytao_request_write_authorization` 拿到自检可通过的 `@我 顺延「吃席」到 wkxk`** —— 第 2 轮已修掉「只在测试里成立」的问题（测试不再硬编码 writes_allowed） |
| 把吃席的编码调到赤溪前面 | ❌ | ❌ | 同上 |

A2 验收句式（含未在 21 行内的三条）全部通过：
「确认顺延：吃席 → wkxk，赤溪顺延」「确认执行顺延：吃席 → wkxk，赤溪 → wkxkv」
「执行顺延：吃席 wkxk，赤溪 wkxkv」「执行顺延 吃席 wkxk 赤溪 wkxkv」「执行顺延吃席wkxk赤溪wkxkv」
——全部 ✅；「把吃席的编码放在赤溪前面」按规格允许拦截，但必须给出自检可通过的建议指令，已满足。

### 误授权（hazard）扫描

30 条日常闲聊 + 问句 + 否定 + 引用改写（含核验报告点名的「提前告诉我结果 / 插入一张图片看看 /
占用率是多少 / 放到明天再说 / 调到静音模式 / 排在我后面的是谁 / 放在这里就行 / 执行 /
执行完了记得删除备份 / 会议顺延到下周 …」）→ **0 条被误判为 AUTHORIZED**；
11 条必须授权的句式 → 全部授权。
（注：裸「顺延」在**修复前**就已经 auth=True，由绑定层拦下，非本次引入，故不计入 hazard。）

---

## 3. 与规格的偏差 / 需要负责人知晓

1. **「执行」的处理已按评审改正（原偏差撤回）。** 第 1 轮把「执行」加进
   `_EXPLICIT_REQUEST_PREFIX_RE`，并声称它「只对已过授权视图的子句生效，不可能抬升闲聊」——
   **这个论断被评审实测证伪**：授权视图除了 positive command 之外还收 `is_protection_clause`
   的子句，而「保留」同时在 `_PROTECTED_WORD_RE` 和 `_MUTATION_INTENT_RE` 里，于是
   「执行 … 保留 …」形态的闲聊会被抬成 AUTHORIZED（8/8 反例）。
   现改为：`_EXPLICIT_REQUEST_PREFIX_RE` 撤回「执行」，改在 `message_authorizes_mutation`
   末尾**先用 `_COMMAND_PREFIX_RE` 剥前缀、再判动词是否在 0 位**。
   `_MUTATION_INTENT_RE` 仍一字未动。
2. **A4 比规格更保守一档。** 规格说「引号内内容命中动词表时不得抹掉整段」，实现为
   「引号内是**词条**则保留、是**整条命令**（含操作对象或超长）则仍 redact」，
   否则「添加「删除草稿条目 12」 aa」会让模型把引号里的 12 绑成删除目标（已加负例测试）。
3. **建议指令的门槛不再是 `writes_allowed`。** 评审的 major-4 与 major-5 互相拉扯
   （既要在只读轮给得出建议，又不能把现成授权递给只是提问的人）。统一解法：门槛改成
   `message_requests_change(当前消息, 工具, 参数)`——用户本轮确实在要求**这一类**改动才给，
   问句/解释/否定/取消一律不给，动作类别不匹配（说加词却要删条目）也不给。
   注入场景下用户原话没有改动意图，因此拿不到任何现成指令。
4. **禁区遵守**：`_MUTATION_INTENT_RE` 未改；`_stage_agent_mutation` 死代码未删；
   未改 keytao-next；未 commit / 未 push。

## 4. 评审 major-1 的残留一例（如实说明）

「执行保留策略」在修复后仍为 `True`。它**不是本次引入的新类**：
`保留策略`（去掉「执行」前缀后的原句）在 HEAD 上本来就是 `True`
（「保留」是 `_MUTATION_INTENT_RE` 成员且位于 0 位）。剥前缀后两者判定必然相同，
这一点已用等价性测试 `test_stripping_a_command_prefix_adds_no_new_authorization_class` 固化。
要消掉它必须把「保留」移出 `_MUTATION_INTENT_RE`，属于禁区，未做。
评审点名的另外 7 句已全部回到 `False`。

## 5. 评审发现逐条处置

| 编号 | 处置 | 证据 |
|---|---|---|
| **major-1** 「执行」扩权 | **已修**（见 §3.1） | 8 句 hazard 7 句消除、第 8 句与 HEAD 等价并有等价性测试；A2 全部验收句式 + 15 条必须授权句式仍全 True |
| **major-2** word_discovery 死代码 | **已真修** | `keytao_json(allow_status={400})` + 读 400 体拿快照再确认。按真实服务端语义回放：链路 5 步，`confirmed batch-draft writes reached the server: 1`（评审实测为 0）；测试链路断言同批改为真实 preview→confirm |
| **major-3** 部署顺序耦合 | 属 keytao-next / 发布流程，本工作流未改；已在 §6 记明「先发 bot 再发 next」 | — |
| **major-4** 只读轮拿不到建议 | **已修** | 新增只读轮 `keytao_request_write_authorization`（不写数据），仅在本轮确有改动意图时挂出；4 条编排级测试覆盖 |
| **major-5** 现成授权递给提问者 | **已修** | 建议门槛改为 `message_requests_change`；`test_a_question_never_receives_a_ready_made_authorization` 覆盖 5 例 |
| **minor-6** batch_id 可自由指定 | **已修** | 模型发起的锚定读取只允许本轮服务端返回过的 batchId（`untrusted_batch_reference`），参数写进 schema；内部锚定调用不受影响 |
| **minor-7** 去重键太粗 | **已修** | 键改为 (blockReason, 工具名, 参数指纹) |
| **minor-8** 无草稿顺延硬失败 | **已修**（本轮解锁） | 按 CAS-on-absence 契约：基线 `("", 0)` 进 digest、写入传 `batch_id=None`+`version 0`；`test_shift_phrase_code_works_with_no_draft_batch`（9 checks）+ 编排级一次授权即执行测试 |
| **minor-9** 不可达守卫 | **已删** | 连同 minor-8 一起处理，并改成「调用方已有基线就不再重新解析批次」这一有实际作用的逻辑 |
| **minor-10** A3 覆盖过薄 | **已修** | 两半各补敏感用例；变异复验：撤销空白保留 → 3 条测试红（含新头牌），邻接过滤回退 → 2 条测试红 |
| **minor-11** 保护词被整句吞掉 | **已修** | 只忽略用户用引号点名的那一处；「把「保留」顺延到 wkxk，保留原编码」仍被拦 |
| **note-12** 注释与预处理不符 | **已改** | 注释改为说明生产链路已剥离 @，该正则存在的意义是让 `@我 …` 建议按同一套校验器自检 |
| **note-13** 写路径未接选择器 | keytao-next 侧，本工作流未改 | — |
| **note-14** 多余未锚定读 | **已改** | 未锚定读只剩「预览兼指针探针」这一次，指针一致时其结果即所需；条目读取在有锚点时始终锚定 |

## 7. 第 3 轮：复核轮两处残留的处置

| 残留 | 处置 | 实测证据 |
|---|---|---|
| **major-4/5 残留**：建议指令只校验动词类别，不校验操作数；位置词让闲聊过门槛 | **已修**：①`self_checked_suggested_command` 增加操作数在场校验——建议里要提到的每个实体必须已出现在用户原话中；②`_POSITIONAL_CHANGE_RE` 移除「占用\|提前\|前面\|后面\|位置」 | 评审矩阵复现：合法 **5/5 全保**（含事故第 1 句、「把吃席顺延一下」、「删掉草稿条目 12」、「提交草稿吧」、「想撤回一下」）；滥用 **3/3 全挡**（事故原句 + 模型自选参数 → 无建议）；闲聊 **10/10 全挡** |
| **minor-6 残留**：未声明 `batch_id` 经写工具结果回显洗进可信集，两跳后放行锚定读 | **已修**：①可信批次校验扩到**所有**带 `batch_id` 的模型调用（第一跳即被拦，写也不会打到陌生批次）；②`_collect_trusted_batch_ids` 不再信任「调用方自己传进来的那个 id」（即便结果回显也不收） | 评审 PoC 逐跳复现：step0 拦、**step1 拦**（原为 reached tool）、trusted 集保持空、step2 仍拦 → `CLOSED` |

两处修复都补了负例测试（见 §1 测试表最后两行）。另做变异复验：把「自确认前先收集可信批次」这一步删掉，
`test_bound_shift_executes_without_asking_for_a_ticket` 与 `test_a_saved_shift_ticket_carries_the_server_plan`
立即变红——说明合法自确认路径确实被测试覆盖，不是靠放宽守卫换来的。

**未处理（评审列为 P2，本轮未纳入）**：转述句「他说顺延吃席到 wkxk」仍可能换到建议指令
（`message_requests_change` 不做转述过滤）；不过第 3 轮的操作数校验已要求「吃席」确由用户本轮打出，
社工面进一步收窄到「用户自己打出了该词条 + 该类动词」的转述句。

## 6. 遗留 / 建议后续

1. **部署顺序（major-3）**：必须**先发 keytao-bot、再发 keytao-next**。
   反向顺序下，旧 bot 遇到新 next 的 `batchId:null` 会把「加词」直接判失败，
   而「草稿全部提审完」是极常见状态。
2. **`delete_actor` 仍被清空路径使用**：清空会删掉全部草稿条目，任何草稿域票据都确实失效，
   故只改了撤回路径（A6 要求的范围）。
3. **A6 的路径选择理由**：主路径选 orchestrator 服务端自确认，而非给 shift schema 暴露 digest——
   自确认是确定性的（不依赖模型是否愿意回传 digest），且第二次调用完整重跑绑定校验 +
   服务端 CAS，安全性不降；票据链路的 digest 合并也一并修好，真有服务端风险警告时从
   「两张票」降为「一张票」。
4. 报告里的 P2 项（引用确认 digest、`[不可信参考资料]` 拆成独立消息、`build_reply_context`
   无条件注入、`confirmed` schema 自相矛盾）不在本次范围，仍待处理。
