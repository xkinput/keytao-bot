# keytao-bot 授权/状态机修复规格（Workstream A）

## 必读背景（先完整读完再动手）

1. 调查报告：`scratchpad/keytao-bot-authz-findings-claude.md`
2. 交叉核验报告（对调查报告的修正，**以此为准**）：`scratchpad/keytao-bot-authz-verify-claude.md`
3. 事故时间线简报：`scratchpad/keytao-bot-authz-investigation-brief.md`

（scratchpad = /private/tmp/claude-501/-Users-rea-code-keytao-org/b793e2c6-2741-4246-a713-f813b130754a/scratchpad）

目标仓库：/Users/rea/code/keytao-org/keytao-bot（干净工作区，改动直接落在工作区，**不要 commit**）。
仓库内有 .venv 可跑测试。报告中的 file:line 以当前代码为准，若有漂移以语义定位。

## 总目标

用户的明确当前指令一次授权即执行；拦截提示的原因必须真实、补救指令必须自检可通过；
同一原因一轮最多提示一次；状态机不弄丢词条。安全底线不降：注入防护依赖的
writes_allowed/来源分级/服务端 CAS 语义全部保持。

## 改动清单（按此顺序实现，每完成一项跑一次相关测试）

### A1. 结构化拦截原因 + 诚实文案（P0）
- 位置：`keytao_bot/harness/tools.py` `_validate_policy`（~1055-1064 的写死文案分支）。
- 拦截结果改为结构化：`blockReason`（枚举：verb_not_matched / binding_incomplete /
  source_untrusted / ticket_required / …按实际分支梳理）+ `missing`（缺什么）+
  `suggestedCommand`（一条**先经同一校验器自检确认能通过**才允许给出的建议指令，自检不过就不给建议、只说明原因）。
- 删除「历史、记忆、引用或附件内容不能授权修改草稿」这句与纯文本路径真实原因脱钩的统一文案；
  附件通路保留其准确的那半句。每个 blockReason 一句对应的真实文案。
- 所有补救建议文案必须带「@我 」前缀（否则群聊消息会被 should_handle 丢弃，见核验报告）。
- 无任何现有测试断言旧文案（核验已确认），可安全替换。

### A2. 绑定层词表补齐（P0，主因）
- `_WORD_RIGHT_SUFFIXES` 增加「顺延」「的编码」等事故句式所需后缀；「执行」加入 `_COMMAND_PREFIX_RE`（核验实测安全）。
- **红线**：扩表只进绑定层（`_ACTION_TOKENS["Change"]`、`_WORD_*` 前后缀），
  **禁止**动 `_MUTATION_INTENT_RE`；**禁止**加入「提前/占用/排在/放到」这类日常词
  （核验实测它们会把闲聊误判成 AUTHORIZED）。
- 验收：事故中用户实际发送过的句式全部能通过意图+绑定，至少包括：
  - 「确认顺延：吃席 → wkxk，赤溪顺延」
  - 「确认执行顺延：吃席 → wkxk，赤溪 → wkxkv」
  - 「执行顺延：吃席 wkxk，赤溪 wkxkv」
  - 「执行顺延 吃席 wkxk 赤溪 wkxkv」
  - 「执行顺延吃席wkxk赤溪wkxkv」
  - 「把吃席的编码放在赤溪前面」→ 此句无编码，允许结果为「意图通过+绑定不完整」，
    但拦截输出必须给出自检可通过的建议指令。

### A3. Tokenization 修复（P0，次因）
- `tools.py:395`：不再无差别删空格（改为保留 token 边界的规范化）。
- `tools.py:730-731` + `_span_distance`：紧邻目标词的编码 token 不得被距离过滤丢弃。
- 保护集必跑且必须保持绿：`test_memory_safety.py` 2267-2275 / 2287-2302 / 2337-2348
  对应的测试（按测试名定位，行号可能漂移）。

### A4. 引号反噬修复（P0，新根因）
- `trusted_mutation_source` ~378 行：「」内内容命中动词表时不得抹掉整段/导致授权失败；
  引号内内容应视为词条字面量参与绑定。
- 回归测试：`把「保留」顺延到 wkxk`、`把「提交」改成 abcd`、`把「修改」删除` 均能正确授权并绑定。

### A5. 只读轮不暴露写工具（P0，新根因，断根「反复拦截」）
- `keytao_bot/harness/orchestrator.py` ~179-181：工具清单在 visual_context 之外增加按
  `mutations_allowed` 过滤；只读轮不给模型任何写工具。
- 回归测试：mutations_allowed=False 时工具清单里无写工具。

### A6. 一次授权即执行（P1）
- 目标：绑定完整、来源可信的当前指令，一次确认即可完成顺延（不再 local_preview→server_warning 两段两票）。
- 方案（核验已验证安全性）：shift 工具 schema（`keytao_bot/skills/keytao-draft/tools.py` ~3905-3921）
  暴露 planDigest 等参数，或 orchestrator 做服务端自确认——`orchestrator.py` ~996-999 不再丢弃
  服务端 digest。任选一条主路径实现，说明理由。`test_state_machine` 固化的是服务端 CAS，
  暴露参数不降安全性。
- `openai_chat.py` ~7209 `delete_actor`：改为按关联性作废票据（撤回批次只作废与该批次相关的票据），
  不再无差别清空。

### A7. 草稿读取按 batch_id 锚定（P1）
- `openai_chat.py` ~5916 与 `harness/tools.py` ~855 `_fetch_draft_snapshot` 的无 batch_id 读路径：
  会话上下文已知 batch_id 时必须带 batch_id 查询；锚定结果与 latest-draft 不一致时，
  回复中必须明确告警（「当前草稿指针与上次操作的批次不一致」）。

### A8. 同一原因一轮最多提示一次（P1）
- 同一 blockReason 在同一轮对话内重复触发时，第二次起只给一句简短提示，不再输出完整补救长文。

### A9. 新增回归测试
- 顺延正例集（A2 验收句式全部进测试）。
- 引号词条正例（A4）。
- 只读轮工具清单（A5）。
- 撤回后草稿可见性：bot 侧以 mock 服务端方式断言「撤回批次 X 后，草稿读取能看到 X 中的条目」（配合 A7 锚定）。

## 禁区

- 不动 `_MUTATION_INTENT_RE` 的语义范围（只能修 bug，不能扩权）。
- 不删 `_stage_agent_mutation` 死代码（P2 留待后续）。
- 不动与本任务无关的测试语义；`test_memory_safety.py` 2455-2461（agent 绑定完整变更直接写入）
  是本次修复的方向锚点，必须保持绿。
- 不 commit、不 push、不改 keytao-next（那是另一个并行工作流的范围）。

## 验收

1. 全量跑 `test_memory_safety.py`、`test_state_machine.py`、`test_security_fixes.py`、
   `test_review_gate.py`、`test_llm_policy.py` + 新增测试，报告逐套通过数。
2. 用报告里的 21 行矩阵句式做一次前后对比（修复前 5/21），报告新通过率及仍不通过句式的原因。
3. 产出改动摘要（文件 → 改动点 → 对应规格编号）写到
   scratchpad/keytao-bot-fix-summary-A.md，最终回复给要点。
4. 中途若因 API 中断被续跑，先 `git -C /Users/rea/code/keytao-org/keytao-bot diff --stat` 审查半成品再继续，禁止盲目重写。
