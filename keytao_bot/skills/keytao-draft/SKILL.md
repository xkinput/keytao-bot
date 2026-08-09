---
name: keytao-draft
description: 键道输入法草稿批次管理工具。添加词条、查看草稿列表、删除草稿条目、提交审核。关键词：添加、创建、删除、查看、撤销词条，提交审核，批量加词。使用前需检查用户是否已绑定账号。
---

# 键道草稿批次管理工具

管理用户的键道词条草稿：添加、查看、移除条目，并在就绪时提交审核。

## 使用前提

⚠️ 用户必须先绑定键道平台账号（`/bind`），否则所有工具调用都会返回 404。

## 工具一览

| 工具 | 用途 |
|------|------|
| `keytao_batch_add_to_draft` | **批量**添加多个词条到草稿 |
| `keytao_batch_remove_draft_items` | **批量**从草稿中删除词条（按 ID） |
| `keytao_create_phrase` | 添加 / 修改 / 删除单个词条 |
| `keytao_get_batch_preview` | 查看草稿 diff 预览（含上下文行，**查看草稿时优先用此工具**） |
| `keytao_list_draft_items` | 查看草稿条目列表（含 ID，用于删除操作） |
| `keytao_remove_draft_item` | 从草稿中移除指定条目 |
| `keytao_update_draft_item_weight` | 调整草稿中唯一已知词条的权重（先读取草稿） |
| `keytao_recall_batch` | 撤回最近一次提审，恢复为草稿 |
| `keytao_submit_batch` | 提交草稿批次进行审核 |

## 工具详解

### keytao_batch_add_to_draft（批量首选）

用户一次提交多个词条时**必须优先使用此工具**（3 个及以上词条）。

**AI 无需传递的参数**（自动注入）：`platform`、`platform_id`

**需要传递的参数：**
- `items` (list): 词条列表，每条包含：
  - `word` (str): 词条内容（Change 操作时为**新词**）
  - `code` (str): 键道编码
  - `action` (str, optional): 默认 `Create`；`Change` = 修改已有词条的词文本；`Delete` = 删除
  - `old_word` (str, **Change 操作时必填**): 旧词条内容。不传会被后端拒绝并报错"修改操作需要指定旧词"
  - `type` (str, optional): 词条类型；用户明确指定类型时必须传。声笔笔=`CSS`，声笔笔单字=`CSSSingle`，词组=`Phrase`，单字=`Single`，补充=`Supplement`，符号=`Symbol`，链接=`Link`，英文=`English`
  - `remark` (str, optional): 备注

**⚠️ Change 操作：old_word 的来源**

**⚠️ Change/Delete 操作：type 的来源**

如果用户说“修改声笔笔/声笔笔单字/词组/单字/补充/符号/链接/英文类型下的编码”，必须把对应 `type` 写入每条 item。不要省略 type；省略会按词组处理，可能改到错误词库。

`old_word` 是词库中现有词条的内容（修改前的词）。根据用户提供的信息不同，取值方式不同：

**情况 A：用户已提供旧词（最常见）**

用户消息中直接列出了"旧词 编码"对（如 `防粘 fpnm`），则：
- `old_word` = 消息中的旧词（如 `防粘`）
- `word` = 对旧词做替换后的新词（如 `防黏`）
- `code` = 消息中的编码（如 `fpnm`）

⛔ **严禁**在此情况下调用 `keytao_lookup_by_codes_batch`、`keytao_lookup_by_word`、`keytao_encode` 等任何查询/编码工具。用户已提供所有信息，直接构建并调用 `keytao_batch_add_to_draft` 获取完整只读预览：
```python
keytao_batch_add_to_draft(items=[
  {"action": "Change", "old_word": "防粘", "word": "防黏", "code": "fpnm"},
  {"action": "Change", "old_word": "胶粘", "word": "胶黏", "code": "jcnm"},
  # 一次性包含所有词条，不拆分，不做任何预查询
])
```

**情况 B：用户只提供编码，未提供旧词**

此时才需要先 `keytao_lookup_by_codes_batch([codes...])` 查出当前词，再用查询结果的 `word` 作为 `old_word`。

**特性：**
- 遇到硬冲突（词条+编码完全重复、要删除的词不存在等）→ 跳过，记入 `failed`
- 首次调用（包括没有警告时）只返回完整预览，不写入；若当前本人已经用包含完整词条/编码的指令，或通过原生引用候选消息回复短指令，授权了完全相同的目标，程序可在同一操作中自动回放精确预览票据，不再要求用户输入验证码
- 确认写入必须绑定服务端返回的 `batchId`、`contentVersion` 和 `warningDigest`；快照变化时拒绝执行
- 草稿中已存在的相同操作 → 跳过，记入 `skipped`
- 最终返回 `successCount`、`failedCount`、`skippedCount`、`warnedCount`、`failed[]`、`skipped[]`、`warned[]`、`draftItems[]`

**返回值示例：**
```json
{
  "success": true,
  "message": "成功写入 2 条，重码警告 2 条",
  "batchId": "uuid",
  "successCount": 2,
  "failedCount": 0,
  "skippedCount": 0,
  "warnedCount": 2,
  "failed": [],
  "warned": [
    {"index": 0, "word": "左利手", "code": "zle", "reason": "编码 \"zle\" 已被词条 \"自来水\" 占用，将创建重码（建议权重: 101）"},
    {"index": 1, "word": "右利手", "code": "yle", "reason": "编码 \"yle\" 已被词条 \"原来是\" 占用，将创建重码（建议权重: 101）"}
  ],
  "draftItems": [...],
  "draftTotal": 3
}
```

**`warnedCount > 0` 时：执行「编码警告确认协议」**（见下方独立章节）。不得静默删除、改码或重写用户已经选择的词条；如条目已写入，展示重码事实即可。`failed[]` 非空时，在回复中追加「❌ 冲突条目（未写入）」段落。

### keytao_create_phrase

**需要传递的参数：**
- `word` (str): 词条内容
- `code` (str): 键道编码（纯字母 a-z）
- `action` (str, optional): `Create`（默认）/ `Change` / `Delete`
- `old_word` (str, **Change 操作时必填**): 旧词条内容，不传后端会拒绝。若用户已提供旧词直接填入；若未提供则先调 `keytao_lookup_by_code(code)` 查出当前词
- `type` (str, optional): 词条类型；用户明确指定类型时必须传。声笔笔=`CSS`，声笔笔单字=`CSSSingle`，词组=`Phrase`，单字=`Single`，补充=`Supplement`，符号=`Symbol`，链接=`Link`，英文=`English`。不传则自动推断：
  - 1 个汉字 → `Single`；英文字母 → `English`；链接 → `Link`；符号 → `Symbol`；其余 → `Phrase`
- `remark` (str, optional): 备注
- `confirmed` (bool, reserved): 仅供程序使用保存的精确服务端票据执行；模型不得传入

**返回值示例：**
```json
// 首次完整预览（尚未写入）
{"success": false, "requiresConfirmation": true, "batchId": "uuid", "contentVersion": 3, "warningDigest": "...", "message": "..."}

// 冲突（不可绕过）
{"success": false, "conflicts": [...], "message": "存在冲突"}

// 重码警告（词条尚未写入）
{"success": false, "warnings": [...], "requiresConfirmation": true, "message": "存在警告"}
```

**收到 `requiresConfirmation: true` 时，执行「编码警告确认协议」：**
- 对已由当前消息精确授权的单条 `Create`，若警告仅为 `duplicate_code` / `code_chain_priority` 且警告条目的词和编码与本次目标完全一致，程序自动回放一次服务端 `batchId`、`contentVersion`、`warningDigest` 票据，并在成功回复中展示警告与同码顺序。
- 对已由当前消息逐项精确授权的批量 `Create`，只有全部条目完全绑定、警告为空或仅为上述两类且逐条对应本批目标时，程序才自动回放一次同样的服务端票据。
- `multiple_code`、`skipped_candidate_slot`、批次状态变化、版本冲突、目标不一致或任何未知警告不得自动确认，继续按精确票据等待本人确认。
- 模型不得自行重新调用本工具或构造 `confirmed`、版本、摘要或权重参数。
- 禁止在用户未表态时擅自换到其他编码；若用户明确要求“自动安排空位”，才可使用 `keytao_encode` 的候选链选择第一个空位。

### keytao_get_batch_preview（查看草稿时优先使用）

获取当前草稿的 diff 预览。**AI 无需传递任何参数。**

**与 `keytao_list_draft_items` 的区别：**
- `keytao_get_batch_preview`：返回 unified diff（含上下文行），适合展示"变更效果"
- `keytao_list_draft_items`：返回含 ID 的条目列表，适合执行删除操作

**用户查看草稿时，必须先调用 `keytao_get_batch_preview`；若用户还需要按 ID 删除某条，再追加调用 `keytao_list_draft_items`。**

**返回值：**
```json
{
  "success": true,
  "batchId": "uuid",
  "batchUrl": "https://...",
  "summary": {"added": 4, "modified": 0, "deleted": 2},
  "diff_text": "diff Phrase  bkxfu, bkxfuu\n@@ -10,7 +10,9 @@\n ..."
}
```

**查看草稿时，必须同时调用 `keytao_get_batch_preview` 和 `keytao_list_draft_items`**，将两者合并展示。

**回复格式：**
```
+4 新增  ~0 修改  -2 删除

```diff
diff Phrase  bkxfu, bkxfuu
@@ -10,7 +10,9 @@
 彼岸    bkxf        100
-猛狂    bkxfu       100
+币安    bkxfu       100
+猛狂    bkxfuu      100
 笔形    bkxg        100
` ``

当前草稿（共 N 条）：
• 新增 币安 → bkxfu（权重: 100）
• 新增 猛狂 → bkxfuu（权重: 100）

草稿地址：https://...

发送「提交」以提交该草稿
```

格式规则：
- `diff_text` 放在 ` ```diff ``` ` 代码块中（Telegram 等平台会显示语法高亮）
- summary 行在代码块前面
- diff 代码块之后，紧接着展示 `keytao_list_draft_items` 返回的草稿条目列表（格式同 `keytao_list_draft_items` 的回复格式，但无需重复 summary 行）
- 列表之后附草稿地址
- 最后一行始终为：`发送「提交」以提交该草稿`
- 若 `diff_text` 为空（草稿内容不影响词库），跳过代码块，直接展示列表，并提示"变更预览暂无数据，可能是新词尚未命中规则"
- 若 `success: false`，直接显示 `message`

---

### keytao_list_draft_items

查看用户当前草稿批次的全部条目。**用于获取条目 ID 以执行删除操作，或查看冲突警告详情。**

**AI 无需传递任何参数。**

**返回值示例（含冲突警告）：**
```json
{
  "success": true,
  "batchId": "uuid",
  "count": 2,
  "items": [
    {
      "id": 410, "action": "Change", "word": "赛百味", "oldWord": "四步舞", "code": "sbw",
      "weight": null, "conflictInfo": null
    },
    {
      "id": 411, "action": "Create", "word": "四步舞", "code": "sbwii",
      "weight": 100,
      "conflictInfo": {
        "hasConflict": false,
        "impact": "编码 \"sbwii\" 已被词条 \"赛百味\" 占用，将创建重码（建议权重: 100）"
      }
    }
  ]
}
```

**回复格式（严格遵循）：**

返回值中包含 `summary` 字段（`added`/`modified`/`deleted`），**必须在列表前显示统计行**：

```
+3 新增  ~0 修改  -1 删除

草稿共 N 条：

1. [修改] 四步舞 → 赛百味 @ sbw
2. [新增] 四步舞 → sbwii（权重: 100）
   ⚠️ 编码 "sbwii" 已被词条 "赛百味" 占用，将创建重码（建议权重: 100）

[草稿链接]
```

summary 格式规则：
- `+{added} 新增  ~{modified} 修改  -{deleted} 删除`
- 三项都显示，即使为 0

规则：
- action 对应显示：Create → 新增，Change → 修改，Delete → 删除
- Change 操作格式：`[修改] oldWord → word @ code`
- 如果 `weight` 不为 null，在编码后加「（权重: N）」
- 如果 `conflictInfo.impact` 不为空，在该条目**下一行**加「⚠️ {impact}」

**空草稿时：**
```
草稿里还没有内容哦，可以用"帮我加个词"开始添加～
```

**工具返回 success=false 时：**
直接告诉用户具体 message 字段的内容，不要自己发挥或说"不支持"。

### keytao_remove_draft_item

从草稿中删除指定条目。**需要先调用 `keytao_list_draft_items` 获取条目 ID。**

**需要传递的参数：**
- `pr_id` (int): 条目的数字 ID（来自 list 返回的 `items[].id`）

**AI 无需传递**：`platform`、`platform_id`（自动注入）

### keytao_batch_remove_draft_items

批量从草稿中删除多个条目。**需要先调用 `keytao_list_draft_items` 获取条目 ID 列表。**

**需要传递的参数：**
- `ids` (list[int]): 要删除的条目 ID 列表

**AI 无需传递**：`platform`、`platform_id`（自动注入）

**特性：**
- 一次调用删除多条，效率高于逐条删除
- 不存在的 ID、无权限的条目会在 `failed` 中说明原因
- 返回 `successCount`、`failedCount`、`deleted`（含 word/code）、当前草稿快照

**AI 回复格式：**
```
已从草稿删除 N 条：
• [action] word（编码：code）
...
（如有 failed）以下条目删除失败：id - 原因
[草稿链接]
```

**使用时机：** 用户要删除 2 个及以上草稿条目时，优先使用此工具。

### keytao_recall_batch

撤回最近一次提审，将批次从"审核中"状态恢复为草稿。首次调用无需参数，由程序获取精确批次和版本的只读快照，再在同一条当前明确指令内回放该快照；用户无需抄写确认票据。若批次或版本已经变化，必须停止并报告，不得改用新的目标重试。

⚠️ **仅当用户明确说"撤回"、"撤销提交"、"取消提审" 时才调用。**

用户明确说“撤回提交并清空草稿”时，先精确撤回该批次，再只清空该批次恢复出的草稿条目。两步都使用服务端版本和目标摘要做 CAS 校验；若撤回成功但清空失败，必须如实报告部分完成状态。

若当前已存在有内容的新草稿，API 会返回 `success: false`，直接告诉用户 `message` 内容即可，无需额外解释。

---

### keytao_submit_batch

提交草稿批次进行管理员审核。**AI 无需传递任何参数。**

⚠️ **仅当用户明确说"提交"、"提审"、"发起审核"、"submit" 时才调用（初次调用 `confirmed` 不传或 false）。**

`keytao_submit_batch` 首次调用始终返回完整只读快照。程序会保存其中的 `batchId`、`contentVersion`、`snapshotDigest`、`warningDigest` 和 `auditDigest`。若用户当前指令已明确要求“提交”，或已通过完整词条/编码、原生候选引用明确授权“加入并提交”，且快照没有增加未授权条目，程序直接在同一操作中回放；否则只询问一次。模型不得自行构造 `confirmed=True` 或摘要字段。

提交确认复用首次检查保存的审计快照，不再次调用 LLM 改写结论。快照按用户、批次、版本和摘要绑定，单次 claim 后不可重放；确认请求超时或结果不确定时只能先“查看草稿”，不得自动重试。缓存有单条和总字节上限，超限时安全拒绝。

⛔ **提交确认的铁律：只能由程序回放保存的 submit_batch 精确票据；模型永远不能自行确认。用户的当前明确提交指令可授权程序完成这次精确回放。**
模型绝对禁止自行调用 `keytao_submit_batch(confirmed=True)`、`keytao_create_phrase`、`keytao_batch_add_to_draft` 或任何其他写工具。

🚫 **提交成功后，严禁再调用任何其他工具**（包括 `keytao_list_draft_items`）。提交后批次已不再是草稿，调用其他工具会立即创建一个新的空草稿——这是错误行为。用户可能还需要撤回本次提交，直接回复提交成功格式即可。

## 编码警告确认协议

**触发条件（任一满足即触发）：**
- `keytao_create_phrase` 或 `keytao_batch_add_to_draft` 首次返回完整预览（无论是否有警告，均**尚未写入**）
- 已授权的单条 `Create` 若仅含目标一致的 `duplicate_code` / `code_chain_priority`，程序自动确认一次并展示警告；本节其余确认规则继续适用于 batch、多编码、跳空位、未知警告和快照冲突
- `keytao_submit_batch` 首次返回提交快照（批次仍为草稿）
- `keytao_recall_batch` 首次返回待撤回批次快照（批次仍未撤回）

### 用户意图与处理

**在执行协议前，必须先判断用户意图：**

| 用户意图 | 典型表达 | 处理方式 |
|---|---|---|
| **明确要求保留当前编码** | "就用这个编码"、"加重码"、"确认重码"、"重码也行" | 展示完整预览；本人发送当前精确票据后由程序写入 |
| **明确要求自动安排空位** | "自动顺延"、"帮我找空位"、"放到第一个空位" | 只使用 `keytao_encode` 返回的候选链选择第一个空位，并说明最终编码 |
| **未表态（默认）** | 普通加词请求，未说明如何处理警告 | 单条 `Create` 的目标一致重码自动确认一次并展示警告；其余警告保持当前选择并询问确认，不换码 |

除单条 `Create` 的目标一致重码/排序警告外，用户明确要求保留重码或确认跳过空位时仍先展示服务端完整预览；确认阶段只能由保存的精确票据执行，模型不得自行重调工具或改动参数。

---

**安全边界：** 用户给出的编码、候选编号和“重码/重新编码”选择都属于修改意图的一部分，模型不得静默替换。跳过更短空位时必须明确提示被跳过的编码；只有用户确认后才能继续。

---

### 自动安排空位（仅在用户明确要求时）

1. 对每个目标词调用：
- `keytao_encode(word)` → 获得 `{ code, altCodes: [altCode0, altCode1, ...] }`，组成有序编码链 `[code, altCode0, altCode1, ...]`
- `keytao_lookup_by_codes_batch([code, altCode0, altCode1, ...])` → 查哪些编码在词库中已有词条

2. 按工具返回顺序找第一个空位作为 `targetCode`，不得自己拼码或沿用别的词的候选链。
3. 在写入前说明 `词 → targetCode（前面的码位为何不可用）`；若这会改变用户此前明确选择，仍需再次确认。
4. 已经写入草稿的条目不得为了“优化”自动删除重建，除非用户明确要求重新编码该条目。

---

## 指定占用编码处理协议

**核心规则：允许顺延，但顺延必须由 `keytao_shift_phrase_code` 计算。**

### 新词相对位置默认规则

当用户用“把 W 放在 Y 前面 / 后面”指定新词相对位置时，先用服务端返回的候选或查词结果确定 Y 所在编码 C，再调用 `keytao_create_phrase(word=W, code=C)`；不要由模型手工改码或构造顺延项。执行器按以下规则确定实际写路径：

- 前面且没有同码标记：W 取得 C，Y 通过 `keytao_shift_phrase_code` 的计划机制沿 Y 自己的候选链顺延。
- 后面且没有同码标记：W 使用其本轮服务端候选链中 C 之后的首个空位，现有词不移动；没有已核验空位时询问，不写入。
- 前面 / 后面且明确含同码标记：保留同码权重排序。
- 同码标记的精确词表为：`同码 / 同编码 / 同代码`，带 `一 / 一个` 的变体，`相同(的)码 / 编码 / 代码`，`码 / 编码 / 代码(保持)相同`，`重码`，以及 `重复(的)码 / 编码 / 代码`。
- 引号、行内代码、引用 / 记录框架里的同码字样不构成同码请求。

**权重规则（唯一口径）：** 数值越小排序越靠前。同一 `(code, type)` 链从该类型基础权重起向上累加，任何条目都不得低于基础值。基础权重为：Single=10、Phrase=100、Supplement=100、Symbol=10、Link=10000、CSS=100、CSSSingle=10、English=100。同码前插时，新词占用目标词当前权重，目标词及其后所有同类型条目统一 `+1`；必须把这组 Create + Change 作为一个服务端预览 / 摘要 / 版本绑定的批次，不得由模型逐条拼接或把新词降到基础值以下。

同码前插的自动回放范围严格限定为：一个 Create 是用户点名的新词，并且唯一一个 Change 只把用户点名的目标占位词 `+1`。如果该目标词后面还有同码同类型条目，虽然它们的 `+1` 是保持权重链合法所必需的机械级联，也仍算扩大了影响范围，必须展示完整计划并等待用户明确确认；不得把整条级联视为“只移动了点名占位词”。自动回放仍须携带服务端 `batchId`、`contentVersion`、`warningDigest`，且最多执行一次。

用户明确说“将草稿中「W」C 的权重调整为 N”或等价变体时，先调用 `keytao_list_draft_items`，再仅以其返回的唯一 `W + C` 调用 `keytao_update_draft_item_weight(word=W, code=C, weight=N)`。词、编码、数值缺一不可。若 N 低于类型基础权重，说明实际基础值并停止；不要钳制到基础值，也不要让用户重发同一句，可建议提高同码同类型的其他词条权重。

只有顺延计划除 Y 外还会移动其他词时，才要求用户通过现有计划票据再明确确认；仅移动 Y 到空位时，程序按计划摘要和内容版本自动回放一次，并在回复中列出 W 与 Y 的最终编码。

用户说“把 A 改到 xxx”或“让 A 用 xxx”时，如果 `xxx` 已被 B 占用，可以把 B 顺延走；但 B 的新编码必须通过 `keytao_encode(B)` 得到 B 自己的候选编码链，再从 B 当前编码之后取下一位。不能把 A 的候选编码链套到 B 身上。

### 必须使用的工具

调用：

```python
keytao_shift_phrase_code(word="会员费", target_code="hyfio")
```

该工具会自动：
- 检查 `target_code` 是否是目标词 A 的有效候选编码
- 若用户指定的是飞键/编码系列，使用 `requestedCodeAnalysis` 判断是否属于固定飞键规则，并返回同系列支持的候选码
- 查询 A 当前旧编码
- 查询目标编码占用词 B
- 对 B 调 `keytao_encode(B)`，从 B 自己的候选链里找下一码
- 如果下一码也被 C 占用，继续对 C 调 `keytao_encode(C)` 并顺延
- 每一步都检查能否继续顺延，直到遇到空码或被 A 旧编码释放的位置
- 清理相关旧草稿条目
- 先返回完整顺延计划和摘要；用户确认后才在同一事务写入 Delete+Create
- 返回 `shiftPlan.shifted`，说明顺延计算了哪些词

### 禁止的操作

- 禁止手工构造顺延 Delete+Create；必须调用 `keytao_shift_phrase_code`
- 禁止用目标词 A 的编码链给被挤词 B 选新码
- 禁止没检查目标码是否空就写入顺延结果
- 禁止顺延后不告诉用户移动了哪些词
- 禁止先批量删除草稿中的大量条目，再按模型规划重建

### 示例

```
词库：会员费@hyfa，换言之@hyfio，换衣服@hyfi
用户：还是会员费改 hyfio 吧，换衣服别动了

→ 调用 keytao_shift_phrase_code(word="会员费", target_code="hyfio")
→ 工具计算：
  1. 会员费可用候选：hyf, hyfi, hyfio, hyfioa；目标 hyfio 有效
  2. hyfio 当前已有「换言之」
  3. 对「换言之」调用 encode，候选为 hyf, hyfi, hyfio, hyfioo
  4. 换言之当前位置是 hyfio，下一位是 hyfioo
  5. hyfioo 为空，可以顺延

工具写入：
  [Delete 会员费@hyfa, Delete 换言之@hyfio, Create 会员费@hyfio, Create 换言之@hyfioo]

回复：会员费已改到 hyfio；顺延计算：换言之 hyfio→hyfioo。换衣服保持 hyfi 不动。
```

---

## 典型场景

### 添加词条

```
用户：帮我加个词：若 ruo
→ 调用 keytao_create_phrase(word="若", code="ruo")
→ type 自动推断为 Single（单字）
```

### 查看草稿

```
用户：我的草稿里有什么
→ 同时调用 keytao_get_batch_preview() 和 keytao_list_draft_items()
→ 展示 summary + diff_text（代码块）+ 草稿条目列表 + 草稿链接 + 提交提示

（若用户还要删除某条，list_draft_items 已经调过，可直接用其返回的 ID）
→ 调用 keytao_remove_draft_item(pr_id=...)
```

### 删除草稿条目

```
用户：把"测试"那条删掉
→ 调用 keytao_list_draft_items() 找到 "测试" 的 id（如 42）
→ 调用 keytao_remove_draft_item(pr_id=42)
```

### 提交审核

```
用户：提交吧
→ 调用 keytao_submit_batch()
→ 成功后回复「批次已提交审核」+ batchUrl，结束，**不再调用任何其他工具**
```

### 提交时遇到重码警告

```
用户：提交吧
→ 调用 keytao_submit_batch()   // 返回 requiresConfirmation: true
→ 回复警告内容，询问是否继续提交

用户：确认
→ 程序消费当前精确票据并回放 batchId、版本和三个摘要；模型不调用工具
→ 成功后回复「批次已提交审核」+ batchUrl，结束
```

### 撤回提审

```
用户：撤回一下
→ 调用 keytao_recall_batch() 取得只读快照
→ 程序在同一条明确指令内按批次 ID + 版本回放
→ 批次恢复为草稿状态，并返回 batchUrl
```

所有草稿新增、删除、撤回和提交的成功或失败回复，只要工具返回了可信 `batchUrl`，都必须把链接带给用户。模型最终回复漏掉链接时，编排层补回一次，不得重复展示。

### 用户指定已占用编码

```
词库：会员费@hyfa，换言之@hyfio，换衣服@hyfi
用户：还是会员费改 hyfio 吧，换衣服别动了

→ 调用 keytao_shift_phrase_code(word="会员费", target_code="hyfio")
→ 工具按各自 encode 链计算顺延
→ 汇报：会员费已改到 hyfio；顺延计算：换言之 hyfio→hyfioo；换衣服保持 hyfi 不动
```

### 编码冲突处理

**`keytao_create_phrase` 遇到目标一致的单条重码：**
→ 程序按精确服务端票据自动确认一次，按原编码加入草稿，并展示与谁形成重码及最终同码顺序；禁止静默改码。

**`keytao_batch_add_to_draft`、多编码、跳空位或未知警告：**
→ 保持待确认状态，展示完整预览并询问是否继续；禁止自动确认或静默改码。

**`keytao_submit_batch` 返回 `requiresConfirmation: true` 时：**
- 已有当前本人明确的提交授权，且快照只含本次授权目标：程序自动按原样回放批次、版本和三个摘要
- 快照含额外旧草稿、目标变化或影响范围扩大：展示完整批次快照、审核结论和警告，只询问一次是否继续
- 模型不得重新调用、换工具或自行构造确认字段

**Delete 操作同样必须先由程序取得精确目标、批次版本和目标摘要。** 当前发送者本轮明确点名了完整删除范围时，这条指令本身授权程序在内部回放同一快照，不要求用户抄写票据；若目标不唯一、服务端返回的范围扩大或包含未点名条目，则只展示一次完整快照并等待确认。成功响应中的 `notes` 字段记录被删除的词条信息：
```json
{
  "success": true,
  "notes": [
    {"index": 0, "word": "卧推", "code": "wltbv", "weight": 0, "type": "Phrase"}
  ]
}
```
收到成功响应后，**必须把 notes 里的删除内容告诉用户**，格式示例：
```
✅ 已加入草稿
• 删除 卧推（wltbv，词组）
```

### 多步操作不中断原则

用户一次要求执行多个步骤（如"先删A再删B再加C"）时：
- **按顺序取得每一步的只读快照并做 CAS 回放**，已由当前明确指令完整授权且目标没有扩大时，不在步骤之间重复询问
- Create/Change/Delete 任一步出现新增风险、歧义或影响范围扩大时，只暂停一次并展示完整目标
- “清空草稿”只允许删除服务端锁定的当前批次全部条目；“撤回并清空”只允许清空撤回返回的同一批次
- 写请求超时或结果不确定时立即停止后续步骤，先让用户查看原批次，绝不自动重试
- 撤回或删除的终态会先持久保存，只有最终回复成功送达后才解除该用户的写入锁；机器人重启后只重放原批次结果，不改选新批次或新条目
- 若服务端始终无法核验原操作，用户在网站确认状态后可发送“放弃不确定操作”解除该用户的安全锁；这条指令本身不执行任何草稿写入
- 全部完成后一次性汇报结果

QQ 或 Telegram 的原生回复若能确认被引用消息由机器人本人发送，单词候选消息可以作为 `word/code` 的目标凭证；当前消息发送者始终是新操作的唯一 actor。用户可直接回复“加入并提交”或“添加并提交”，程序不得恢复、消费或覆盖被引用消息原接收者的 ticket，而应在当前 actor 下建立全新操作。写入前必须调用当前整词审词流程重新验证词语、语境读音、推荐编码和目标码占用状态；任一项变化或读音未解决都拒绝写入并要求重新生成候选。没有有效机器人原生引用时，短句不得执行，用户必须发送包含完整目标的指令，例如“添加 窨茶 xwwso 并提交”。

### 草稿链接（batchUrl）

所有已知精确批次的成功、失败、待确认、部分成功或结果不确定响应都应包含 `batchUrl`，指向该草稿批次的网页：
```
https://keytao.vercel.app/batch/{batchId}
```

尚未物化的 absence-CAS 预览只显示「待确认后生成」，不得展示服务端返回的临时 UUID 链接。

**每次操作后必须在回复中附上草稿链接**：
```
草稿地址：https://keytao.vercel.app/batch/xxx
```

查看草稿时，也在列表末尾附上链接，方便用户直接访问。

`keytao_create_phrase` 和 `keytao_remove_draft_item` 的返回值中包含 `draft_snapshot` 字段：
```json
{
  "success": true,
  "draft_snapshot": {
    "count": 3,
    "summary": {"added": 2, "modified": 0, "deleted": 1},
    "items": [
      {"action": "Delete", "word": "卧推", "code": "wltbv", "type": "Phrase", ...},
      ...
    ]
  }
}
```

**必须在每次操作后展示 `draft_snapshot`，并额外调用 `keytao_get_batch_preview` 获取 diff**，格式：
```
+2 新增  ~0 修改  -1 删除

```diff
diff Phrase  wltbv, wltb
@@ -... @@
 ...
` ``

当前草稿（共 3 条）：
• 删除 卧推 → wltbv
• 删除 我推 → wltb
• 新增 卧推 → wltb

草稿地址：https://...

发送「提交」以提交该草稿
```

规则：
- **每次操作成功后，在回复前调用 `keytao_get_batch_preview`** 获取最新 diff，与 `draft_snapshot` 合并展示
- 回复末尾始终附草稿地址和提交提示：`发送「提交」以提交该草稿`
- 若 `diff_text` 为空，跳过代码块，直接展示条目列表

有警告时（`requiresConfirmation: true`），该次需要确认的条目尚未写入草稿；展示当前草稿并询问用户是否继续：
```
✅ 已执行 X 步，当前草稿（共 N 条）：
• ...

⚠️ 第 Y 步需要确认：编码 "xxx" 已被 "某词" 占用，或跳过了更短空位编码。
确认继续添加吗？
```

## 回复格式

**提交成功：**
```
✅ 批次已提交审核！共 N 条。
本喵审核：该批次需管理员审核。
批次地址：https://keytao.vercel.app/batch/xxx
```
批次中只要有一条审词结论为“需管理员审核”，整批都必须保持人工审核状态；不得用其他通过项或仅编码校验结果覆盖。
（成功后**严禁**继续调用任何工具，不要主动询问是否创建新草稿）

**添加成功：**
```
✅ 词条已加入草稿！
词条：若 | 编码：ruo | 类型：单字
草稿共 3 条，输入「提交」发起审核，或继续添加。
草稿地址：https://keytao.vercel.app/batch/xxx
```

**草稿列表：**
```
+1 新增  ~1 修改  -1 删除

📋 草稿批次（共 3 条）

1. [#42] 新增 词组  测试 → ushi（权重: 100）
   ⚠️ 编码 "ushi" 已被词条 "测辞" 占用，将创建重码（建议权重: 100）
2. [#43] 修改 词组  若 → 弱 @ ruo
3. [#44] 删除 单字  若 @ ruo

可以继续添加、删除条目，或输入「提交」发起审核。
草稿地址：https://...
```

格式规则：
- 使用 `display_label` 字段作为词条内容显示
- 有 `warning` 字段时，在该条目下方另起一行加 `⚠️ [warning内容]`
- 类型用 `type_label` 显示在动作后

**删除成功：**
```
🗑️ 已删除：新增 "测试"（编码：ushi）
草稿剩余 1 条。
```

**冲突：**
```
❌ 添加失败
原因：词条 "测试" 与编码 "ushi" 的组合已存在
```

**重码警告：**
```
⚠️ 重码提示
编码 "ushi" 已被 "测试"（权重 100）占用。
添加 "词试" 后将创建重码，两者都会出现在候选框。
确认继续？
```

## 注意事项

1. 只能删除 **Draft** 状态批次中的条目，已提交的无法撤回
2. `code` 必须为纯小写字母，不含空格
3. 草稿批次描述以「键道助手」开头，与用户手动创建的批次区分
