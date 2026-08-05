# keytao 修复交叉评审简报

你是交叉评审者，与实现者不共享上下文。评审对象是两个仓库的**工作区未提交改动**（HEAD 均未动，直接 `git diff` 即可看到全部改动）。不要修改任何代码——只评审，产出发现清单。

## 必读材料（按序）

scratchpad = /private/tmp/claude-501/-Users-rea-code-keytao-org/b793e2c6-2741-4246-a713-f813b130754a/scratchpad

1. 事故与根因背景：`scratchpad/keytao-bot-authz-findings-claude.md`（调查）与
   `scratchpad/keytao-bot-authz-verify-claude.md`（核验修正，与调查有出入处以此为准）
2. 实现规格：`scratchpad/keytao-bot-fix-spec.md`（Workstream A）、`scratchpad/keytao-next-fix-spec.md`（Workstream B）
3. 实现者自述摘要：`scratchpad/keytao-bot-fix-summary-A.md`、`scratchpad/keytao-next-fix-summary-B.md`
4. 实际代码 diff：`git -C /Users/rea/code/keytao-org/keytao-bot diff`、`git -C /Users/rea/code/keytao-org/keytao-next diff`（含新增文件，用 `git status --short` 找 untracked）

## 评审维度（按优先级）

### R1. 安全语义是否被削弱（最高优先）
- 红线核查：`_MUTATION_INTENT_RE` 是否真的一字未动；绑定层扩表是否漏进了意图层。
- **实现者自报偏差 1**：`_EXPLICIT_REQUEST_PREFIX_RE` 也加了「执行”。对抗性评估：构造句式尝试让
  闲聊/转述/引用内容被误判为授权（如「他说执行顺延吃席wkxk」「如果执行顺延会怎样」「别执行顺延」），
  验证实现者“只对已过授权视图的子句生效、不可能抬升闲聊”的论断。
- suggestedCommand 自检机制：建议指令是否真的先过同一套校验器？有没有给出会被拦的建议的路径残留？
- A6 服务端自确认：第二次调用是否完整重跑绑定校验 + CAS？有没有绕过用户授权直接执行的新路径
  （例如模型在未获用户当前指令时自行触发二次调用）？
- 只读轮摘写工具后，模型点名写工具时的结构化回包是否可能泄露可执行的确认路径？

### R2. 正确性与事故闭环
- 用事故时间线（简报 `scratchpad/keytao-bot-authz-investigation-brief.md` 的 11 条）沙盘推演修复后行为：
  每一步用户消息现在会得到什么？「反复拦截 + 每次新格式」是否根除？「吃席」是否不可能再丢？
- A7 锚定 + B2 选择器的跨仓库一致性：bot 侧锚定语义与 next 侧「有内容草稿优先」语义组合是否有缝隙
  （例如 bot 锚定的 batch_id 被服务端判为非当前草稿时的行为）。
- B1 的 200+batchId:null 协议：bot 侧三处 None 透传适配是否完整、有没有漏掉的调用点
  （实现者自报还有第四处 `keytao_shift_phrase_code` 无草稿时 CAS 语义缺失，已列遗留——确认该遗留的
  影响面并判断是否可接受为遗留）。
- A4 保守偏差（引号内含命令 token 仍 redact）：验证其负例测试合理、正例不受损。

### R3. 测试质量
- 新增测试是不是真语义测试（改坏实现会红）而不是「照抄实现」；抽 2-3 个新测试做变异检验
  （心算或实跑：故意破坏对应实现点，测试应当变红）。
- 被改写的 `security-guards.test.ts` 那条：新断言是否完整覆盖新语义（读不建 + 权限绑定仍在）。

### R4. 工程质量
- 改动是否越界（碰了规格禁区、无关文件）；风格是否与仓库一致；有无死代码/调试残留。
- keytao-next：`pnpm test:unit` / `tsc --noEmit` 可自行重跑验证（**禁止跑 `pnpm test`，会连本机开发库并清空它**）。
- keytao-bot：可在仓库 .venv 重跑测试套件验证实现者报告的数字。

## 产出

发现按严重度分级（blocker / major / minor / note），每条含 file:line、问题描述、失败场景、建议修法。
写到 `scratchpad/keytao-fix-review.md`，最终回复给逐条摘要。没有发现也要明确说「按维度 R1-R4 未发现问题」。
