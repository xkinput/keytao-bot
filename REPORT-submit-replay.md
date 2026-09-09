# S61 — 「加入并提交」只写入不提交，且提问在交付时被吞掉

基线：`main`，`3d3dcbb93645a15b9b8209b379f5428cd980bf18`，初始工作树干净。
本轮只在本地修改和验证，未暂存、提交、推送、部署或访问生产。
**付费模型调用 0；真实 provider 调用 0。未读取生产密钥、真实 `.env` 或 `.e2e_key` 内容；未运行 `pnpm test` 或 `e2e.run`；未连接任何服务器。**

## A — 诊断：这一轮到底发生了什么

事故链路已用 fake tools 逐字复现（见 D 段红灯记录）。按实际执行顺序：

1. `openai_chat.py:5210` 起的 `_stage_execute_pending_state`：`PendingAddWord` + `pending_add_and_submit`、
   没有指定编码、推荐码未被占用 → 不进 `handle_pending_message_core`，而是
   `_schedule_background_draft_operation`（`openai_chat.py:5248`）后台执行
   `_perform_add_to_draft_and_submit`。这是 户晨风／hjfo 走的分支。
2. `chat_commands.py:4576` 的 `_perform_add_to_draft_and_submit`：create 预览 →
   `_create_preview_can_auto_confirm` 通过 → 带 CAS 重放 create → 成功（`pullRequestCount 1`，
   `contentVersion 4`）。与生产日志前两次工具调用一致。
3. 同函数接着调 `_perform_submit_current_draft(preview_only=True, auto_confirm=True,
   authorized_items=[本轮那一条])`。批次在本轮之前已是 `contentVersion 3`，快照里含本轮之外的行，
   `_submit_preview_matches_authorized_items` 精确集合不相等 → **不重放，返回 ticket + 提问文案**。
   **这一步是正确的**：协议要求快照含未授权项时只问一次。生产日志里只有一次 submit，正是因为这里没有重放。
4. `openai_chat.py:3006` 的 `_run_background_draft_operation` 把该 ticket 交给
   `draft_operation_coordinator.mark_awaiting_confirmation`，**不写 `conversation_state_store`**。
   这是既有的、有意的隔离（`test_state_machine.py:12784` 起的用例就断言：后台提交确认挂在操作上，
   会话槽仍归新词所有），并且后续「确认」确实由 `_perform_active_operation_confirmation` 消费。
5. 交付边界 `_prepare_user_facing_reply` → `_enforce_advertised_reply_contract` →
   `_enforce_candidate_reply_contract` 取 record 时**只查 `conversation_state_store`**，
   查不到 → `branch=replace_missing_state state=none bindings=0`（与生产 12:13:46 的 WARNING 完全一致）
   → 提问被替换成「当前没有可验证的可执行操作，本次未写入。」
6. S60 在 `_prepare_user_facing_reply`（`openai_chat.py:2824` 起的 deliveries 段）新增的回执恢复逻辑，
   把这句「本次未写入」重写成「本轮写入结果如下。」，再由 `finalize_draft_receipt` 补上
   「已变更：户晨风 → hjfo」和批次链接 —— 于是用户看到一份自信的、只谈写入、对提交只字不提的回执。

结论：**提问不是被 S60 丢的，是被交付边界"record 只有一个来源"丢的（早于 S60 就存在）；
S60 改变的是这次丢失的可见形态**——把一句明显失败的文案，变成了一份看起来成功的写入回执，
于是用户既没拿到提交，也没拿到问题，还拿到了一个像是成功的回答。

### 对给定怀疑的反驳（有证据）

- **不是 `continue_with_submit_preview`。** create 路径根本不经过它：它只存在于
  `_execute_confirmed_tool` 内部，服务于 `keytao_batch_add_to_draft`、`keytao_shift_phrase_code`
  和通用成功分支。本轮的工具序列（create 预览 → create 确认 → submit 预览）与
  `_perform_add_to_draft_and_submit` 一一对应，与 `continue_with_submit_preview` 的序列不符。
- **不是 `finalize_draft_receipt` 本身。** 它只把可信写回响应投影成变更行与链接；
  S61 fixture 里它输出的每一行都属实（户晨风 → hjfo 确实写入了）。问题是它被用来替换了一句
  本该保留提问的文本。
- **S60 的确切责任函数**：`_prepare_user_facing_reply`（deliveries 段，`openai_chat.py:2824-2849`）
  与它依赖的 `_capture_successful_draft_write_delivery`（`chat_commands.py:4032`，S60 放宽了捕获条件
  并新增 `writeReceipt`）。这两处让"被契约替换掉的失败文案"重新变成写入回执，掩盖了提交缺口。
- **shift 路径在 4c75add 能工作、在 3d3dcbb 仍然能工作**，不是因为它避开了这个洞：
  09:29 那轮的 submit 快照与授权集合相等，直接重放成功，从未需要这张 ticket。
  本轮新增的 shift 用例与既有 `test_s60_scenario` 在 3d3dcbb + 本修复上都绿。

## B — 修复

1. `openai_chat.py:2549` `_active_operation_state_record` + `_enforce_candidate_reply_contract`
   （`:2580`）：会话槽没有记录时，把"等待确认"的后台操作 ticket 投影成同一个 `PendingStateRecord`。
   这是交付契约里**唯一**的 record 取值点，因此所有走这个边界的回复（含
   `_format_active_draft_operation_message` 这类同形状提示）一次性都被覆盖。
   槽里已有记录时行为不变，后台 ticket 不会顶掉新词的候选记录。
2. `chat_commands.py:7398` `submit_snapshot_extra_items` / `:7423` `_submit_scope_notice`：
   提交预览带 `authorized_items` 时，先按"词 → 码"列出本轮之外的行（超过 8 条折叠成计数），
   并说明提交只能整批进行、引用本条回复「确认」一并提交。原 `提交内容：` 全量清单保留。
   该提示在 `MAX_REPLACE_CONFIRMATION_CHARS` 长度闸门**之后**才前置，
   因此原本可确认的批次不会因为多了这段解释而变成「提交内容过长，本次未确认」并丢掉 ticket。
3. `chat_commands.py:7438` `subset_submit_refusal` + 两个路由接入点
   （`openai_chat.py:4402` 后台 ticket、`openai_chat.py:5094` 会话槽 ticket）：
   「只提交 X／仅提交 X」在两种 ticket 存放位置下都被确定性识别，**不调模型、不消耗 ticket**，
   如实回答"提交是整批操作"，并给出「确认」与「查看草稿」。
   触发条件收紧到 `只|仅`（不含命令引导词 `就`），并在 `_is_explicit_draft_submit_request`
   为真时直接放行，保证「就提交吧」「提交一下」这类整批提交指令仍然走原有的 assent 重放。
4. `harness/orchestrator.py:200` `_command_suggestions_match_pending_batch`：
   只读的「查看草稿」不再让同一条回复里被 ticket 支持的「确认」整体失效
   （该函数上方原本就已认定"全是查看命令"的建议集合无需 ticket，本次只是把这条规则用在混合集合上）。
5. `openai_chat.py:2726` `_append_unreported_submit_status`：新不变量，见 C 段。

### 关于「只提交 户晨风」：按规格兜底，并对广告文案提出反驳

`keytao_submit_batch` 只接受 `batch_id` 与 CAS 参数，没有任何逐项／子集参数
（`keytao_bot/skills/keytao-draft/tools.py:2399`）；也没有"把若干草稿行搬到新批次"的接口。
唯一能凑出子集提交的做法是先删掉本轮之外的草稿行——那是破坏别人数据的写操作，不做。
因此按规格给定的兜底执行：如实说明 + 「确认」提交整批 / 「查看草稿」先整理。

**不在提问里广告「只提交 X」**：广告一个 ticket 消费不了的动作，会被交付契约判为 unbacked
（实测 `_advertised_reply_matches_live_record` 对含「只提交 户晨风」的提问返回 False，
提问会被整条替换掉），也违反本仓"广告即存在可执行状态"的核心不变量。
提问因此改为明说"提交只能整批进行，不能只提交本轮的词"。
用户仍然可以直接输入「只提交 X」，会得到上面那句如实回答，且 ticket 保持可用。

## C — 新不变量

`_append_unreported_submit_status`（`openai_chat.py:2726`，在交付边界最后一段调用）：
本轮指令包含「提交」、且本轮存在真实写入回执时，最终交付文本必须满足三选一：

1. 本轮有 `draft_submit` 回执，或文本已含「已提交审核」／「已加入词库」；
2. 该 actor 存在活的、**完整的 server_warning** `keytao_submit_batch` ticket
   （会话槽或后台操作皆可，即提问仍然成立）；
   `_acknowledge_delivered_draft_mutations` 在每次普通写入后种下的 `_recent_own_write`
   本地指针不是提问，不能解除该不变量；
3. 否则追加一句「本轮只写入草稿，尚未提交审核。」

触发条件不再是"消息里出现『提交』"这个裸子串，而是
`parse_pending_assent_phrase(message).submit_after` 且没有否定／取消，
因此「先别提交」「不要提交」不会收到这句免责说明。

该检查位于所有契约替换与回执重建之后，因此不会再被改写；追加的那句不广告任何需要状态的动作。
效果：「加入并提交」这一轮不可能再以"只写入 + 对提交沉默"的回执收尾。

## D — fixture 与红灯记录

`test_s61_submit_replay.py`，12 项，全部 fake tools／fixture，两个意图模型入口设为调用即失败
（`_classify_message_command_intent`、`_classify_simple_word_query_intent`），
`command_intent_for` 在只提交用例里同样设为调用即失败。假服务端实现 CAS：
create／submit 的 confirmed 调用逐项校验 `batch_id`、`expected_content_version` 与三个 digest，
并断言不会重复写入或重复提交。

| 用例 | 覆盖 |
|---|---|
| A create（QQ／Telegram） | 批次只含本轮项 → create 预览、create 确认、submit 预览、submit 确认四次调用；回执含「已提交审核」与唯一平台链接 |
| A shift | `_execute_confirmed_tool` + `_submit_after` → 顺延写入后在同一操作内重放提交 |
| B 后台路径 | 批次已有旧项 → 只问一次；ticket 落在后台操作上且完整；`_enforce_advertised_reply_contract` 原样保留提问；提问列出「旧词 → qtxx」 |
| B 会话槽路径 | shift + 旧项 → ticket 落在会话槽；提问同样被原样保留 |
| B 确认 | `_perform_active_operation_confirmation` 用同一 ticket 完成整批提交，回执含「已提交审核」 |
| B 只提交 | 两条路由都不调模型、不提交、ticket 仍然是同一张 |
| B 整批 assent | 「就提交吧」「确认」对同一张后台 ticket 不被误判为子集请求，`_stage_arbitrate_active_operation` 调度确认后提交真的重放 |
| B 长度边界 | 基础提问长度落在 3300（3250–3500 带内）时，加上范围提示仍然保留 pending_state，不返回「提交内容过长」 |
| C | 写入成功但摘要丢失提交事实时，交付文本追加「尚未提交审核」 |
| C `_recent_own_write` | 会话槽里预置本地写入指针时，不变量照常追加「尚未提交审核」 |
| C 否定 | 「先别提交」的一轮不追加提交免责说明 |

修复前同一 fixture 的红灯（B 用例实际交付文本）：

```text
本轮写入结果如下。
已变更：户晨风 → hjfo
草稿地址：https://keytao.rea.ink/batch/s61-3fc424e6
```

与生产 2026-09-09 12:13 的回执逐字同形（只有批次 id 是 fixture 值）。修复后同一用例交付：

```text
✅ 「户晨风」→ hjfo 已加入草稿。
已变更：户晨风 → hjfo
这批草稿里还有本轮之外的内容，提交会一并提交：
• 旧词 → qtxx
提交只能整批进行，不能只提交本轮的词；引用本条回复「确认」一并提交。
服务端发现需要再次确认的风险
提交内容：
• Create 旧词 @ qtxx（Phrase）
• Create 户晨风 @ hjfo（Phrase）
回复「确认」执行，或「取消」。
草稿地址：https://keytao.rea.ink/batch/s61-3fc424e6
```

A 用例（批次只含本轮项）交付：

```text
✅ 已将「户晨风」→ hjfo 写入草稿，已提交审核。
已变更：户晨风 → hjfo
✅ 批次已提交审核。
草稿地址：https://keytao.rea.ink/batch/s61-3fc424e6
```

## E — 测试尾部（全部在本次改动之后运行，实际尾部）

| 检查 | 实际尾部 | 退出码 |
|---|---|---:|
| `.venv/bin/python test_state_machine.py` | `Results: 2023/2023 passed, 0 failed` / `✅ ALL TESTS PASSED` | 0 |
| `.venv/bin/python test_memory_safety.py` | `Ran 407 tests in 209.369s` / `OK` | 0 |
| `.venv/bin/python test_security_fixes.py` | `Results: 268/268 passed, 0 failed` | 0 |
| `.venv/bin/python test_review_gate.py` | `Results: 443/443 passed` / `✅ ALL TESTS PASSED` | 0 |
| `.venv/bin/python test_llm_policy.py` | `Ran 11 tests in 0.121s` / `OK` | 0 |
| `.venv/bin/python test_word_discovery.py` | `Results: 290/290 passed` / `✅ ALL TESTS PASSED` | 0 |
| `.venv/bin/python -m e2e.test_safety` | `Ran 107 tests in 0.768s` / `OK` | 0 |
| `.venv/bin/python -m unittest test_s61_submit_replay` | `Ran 12 tests in 0.085s` / `OK` | 0 |

因为改到了共享的建议匹配器与交付边界，另跑了相邻场景包：
`test_s61_submit_replay test_s60_scenario test_s60_finalizers test_s60_receipt test_s60_delta
test_s58_advertisement test_s57_options test_s54_multiword test_s57_reorder test_skills`
→ `Ran 113 tests in 0.443s` / `OK`，退出码 0。

S61 fixture 汇总行：`scenario=S61, mode=fixture/fake-tools, paidModelCalls=0, realProviderCalls=0`。

## F — 范围与未验证

- 验证范围仅为静态源码 + 离线 fake 工具 + 规定测试；**没有真实 provider、线上用户、生产服务或生产数据验证**。
- 「服务端发现需要再次确认的风险」是服务端既不返回 `message` 也没有 warnings 时的兜底文案
  （`chat_commands.py` 内 `_format_server_warning_confirmation` 末段），措辞偏内部，但属于既有文案，本轮未改。
- 「只提交 X」只在 ticket 是完整的 server_warning 提交票据时被识别。提交完成后由
  `_acknowledge_delivered_draft_mutations` 写入的 `_recent_own_write` 本地票据不属于此类，
  此时说「只提交 X」仍走原有通用路径；本轮未扩大范围。
- 独立评审复现的三项缺陷（`就提交` 被子集分支吞掉、`_recent_own_write` 解除不变量、
  范围提示把提问挤过长度闸门）与两项 nit（wall-clock `created_at`、裸子串「提交」触发）
  已在本轮修复并各自补了回归；`_active_operation_state_record` 的授权范围按评审结论保持不变
  （同一 ConversationAddress、仅 awaiting_confirmation、只读、CAS digest 仍然校验）。
- 未改动写入目标、确认资格、服务端接口、审批流程或持久化 schema。
- 工作树保持脏：`keytao_bot/harness/orchestrator.py`、`keytao_bot/plugins/chat_commands.py`、
  `keytao_bot/plugins/openai_chat.py`、新增 `test_s61_submit_replay.py` 与本报告，均未提交。
