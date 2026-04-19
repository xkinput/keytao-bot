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
  - `type` (str, optional): 不传则自动推断
  - `remark` (str, optional): 备注

**批量替换字符示例**（将"粘"改成"黏"）：
```python
keytao_batch_add_to_draft(items=[
  {"action": "Change", "old_word": "防粘", "word": "防黏", "code": "fpnm"},
  {"action": "Change", "old_word": "胶粘", "word": "胶黏", "code": "jcnm"},
  # old_word 必须是词库中现有的原词，word 是替换后的新词
])

**特性：**
- 遇到硬冲突（词条+编码完全重复、要删除的词不存在等）→ 跳过，记入 `failed`
- 遇到重码警告 → **自动确认写入**，记入 `warned`，**必须触发后续智能分配流程**
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

**`warnedCount > 0` 时：执行「通用编码自动分配协议」**（见下方独立章节）。协议会删除草稿中用错误编码写入的条目，并以正确 targetCode 重新写入。`failed[]` 非空时，在回复中追加「❌ 冲突条目（未写入）」段落。

### keytao_create_phrase

**需要传递的参数：**
- `word` (str): 词条内容
- `code` (str): 键道编码（纯字母 a-z）
- `action` (str, optional): `Create`（默认）/ `Change` / `Delete`
- `old_word` (str, **Change 操作时必填**): 旧词条内容，不传后端会拒绝
- `type` (str, optional): 词条类型，**不传则自动推断**：
  - 1 个汉字 → `Single`；英文字母 → `English`；链接 → `Link`；符号 → `Symbol`；其余 → `Phrase`
- `remark` (str, optional): 备注
- `confirmed` (bool, optional): 用户明确要求保留重码时传 `True`，让词条以原编码写入

**返回值示例：**
```json
// 成功
{"success": true, "batchId": "uuid", "pullRequestCount": 1, "message": "..."}

// 冲突（不可绕过）
{"success": false, "conflicts": [...], "message": "存在冲突"}

// 重码警告（词条尚未写入）
{"success": false, "warnings": [...], "requiresConfirmation": true, "message": "存在警告"}
```

**收到 `requiresConfirmation: true` 时，执行「通用编码自动分配协议」中的优先级判断：**
- 用户明确要保留重码 → 用原 `code` + `confirmed=True` 重新调用本工具
- 用户未表态（默认）→ 执行自动分配，找到空位编码后用 `targetCode` 重新调用本工具

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

撤回最近一次提审，将批次从"审核中"状态恢复为草稿。**AI 无需传递任何参数。**

⚠️ **仅当用户明确说"撤回"、"撤销提交"、"取消提审" 时才调用。**

若当前已存在有内容的新草稿，API 会返回 `success: false`，直接告诉用户 `message` 内容即可，无需额外解释。

---

### keytao_submit_batch

提交草稿批次进行管理员审核。**AI 无需传递任何参数。**

⚠️ **仅当用户明确说"提交"、"提审"、"发起审核"、"submit" 时才调用（初次调用 `confirmed` 不传或 false）。**

当 `keytao_submit_batch` 返回 `requiresConfirmation: true` 时，告知用户警告并询问是否继续。  
用户确认后（说"确认"、"好"、"是"等）→ **立即重新调用 `keytao_submit_batch(confirmed=True)`**，不要改调其他工具。

⛔ **提交确认的铁律：上一步调的是 submit_batch，确认时只能再调 submit_batch(confirmed=True)。**  
绝对禁止在提交确认时调用 `keytao_create_phrase`、`keytao_batch_add_to_draft` 或任何其他工具——那会把用户带入错误的加词流程。

🚫 **提交成功后，严禁再调用任何其他工具**（包括 `keytao_list_draft_items`）。提交后批次已不再是草稿，调用其他工具会立即创建一个新的空草稿——这是错误行为。用户可能还需要撤回本次提交，直接回复提交成功格式即可。

## 通用编码自动分配协议

**触发条件（任一满足即触发）：**
- `keytao_create_phrase` 返回 `requiresConfirmation: true`（含 `warnings[]`，词条**尚未写入**）
- `keytao_batch_add_to_draft` 返回 `warnedCount > 0`（含 `warned[]`，词条**已写入**但编码冲突）

### ⚠️ 优先级判断：用户是否明确要求保留重码？

**在执行协议前，必须先判断用户意图：**

| 用户意图 | 典型表达 | 处理方式 |
|---|---|---|
| **明确要求保留重码** | "就用这个编码"、"加重码"、"确认重码"、"重码也行"、"直接加"、"强制加" | **跳过本协议**，改用原编码 + `confirmed=True` 直接写入 |
| **未表态（默认）** | 普通加词请求，未提及重码 | **执行本协议**，自动分配至第一个空位编码 |

用户明确要求保留重码时，`keytao_create_phrase` 用原 `code` + `confirmed=True` 重新调用；`keytao_batch_add_to_draft` 的 warned 条目已写入，无需修正，直接按正常成功展示（告知用户该词与哪个词形成重码即可）。

---

**协议目标（仅在用户未明确要求重码时执行）：** 为每个冲突词条找到第一个空位编码（primary → altCode[0] → altCode[1] → ...），直接用该编码入库，不询问用户。

---

### Step 1：查询编码链

对每个冲突词条，并行调用：
- `keytao_encode(word)` → 获得 `{ code, altCodes: [altCode0, altCode1, ...] }`，组成有序编码链 `[code, altCode0, altCode1, ...]`
- `keytao_lookup_by_codes_batch([code, altCode0, altCode1, ...])` → 查哪些编码在词库中已有词条

（多个冲突词条的编码链合并去重后，一次调用 lookup 即可。）

### Step 2：确定 targetCode

在编码链中按顺序找第一个词库里 `phrases == []` 的编码 → 即 **targetCode**。

### Step 3：修正草稿

| 触发来源 | 词条状态 | 操作 |
|---|---|---|
| `keytao_create_phrase` | 尚未写入 | 直接用 targetCode 重新调用 `keytao_create_phrase(word, code=targetCode)` |
| `keytao_batch_add_to_draft` | 已写入原冲突编码 | 从 `draftItems` 找到该词的条目 ID，`keytao_batch_remove_draft_items` 删除，再 `keytao_batch_add_to_draft` 以 targetCode 重新写入 |

多个词条的删除/重写可合并为各一次调用。

### Step 4：回复格式

```
✅ 已写入草稿，自动分配编码 2 条。

📌 编码自动分配：
• 左利手 → zlev（zle 已有「自来水」，顺延至下一空位）
• 右利手 → ylev（yle 已有「原来是」，顺延至下一空位）

当前草稿（共 N 条）：
...

草稿地址：https://...

发送「提交」以提交该草稿
```

- 标题：有分配时写"自动分配编码 N 条"，无冲突时正常写"已写入草稿"
- 每条分配说明：`词 → targetCode（原编码 已有「占位词」，顺延至下一空位）`
- 若 targetCode 与 code 相同（原编码本身就空），不列入分配段落，按正常成功处理

---

## 编码顺序调整协议

**触发条件（必须同时满足，缺一不可）：**
1. 用户当前消息明确指定了**目标词**和**目标编码**（如"我要把跑通加到 pzty"、"让跑通用 pzty"）
2. 该目标编码已被另一个词占用

**与「通用编码自动分配协议」的区别：**
- 自动分配协议：找第一个**空位**放新词（新词往后顺延）
- 本协议：用户**指定了已占用的目标位置**，被挤走的原词才往后顺延

⚠️ 以下情况**不触发**本协议：
- 用户选择了空位编码 → 直接普通加词，无需本协议
- 用户只说了词名，没有指定目标编码 → 先展示候选编码列表
- 用户说"调整顺序/优先级"但没说具体编码 → 先询问目标编码

⚠️ **系统注入指令优先**：如果用户消息中出现 `[系统检测：...]` 提示，必须严格按照提示内容执行，不得自行判断触发条件。

---

### Step 1：查清现状（含草稿冲突清理）

并行调用：
- `keytao_encode(new_word)` → 获取编码链 `chain = [code, altCode0, altCode1, ...]`
- `keytao_lookup_by_code(target_code)` → 查目标位现有词条列表
- `keytao_list_draft_items()` → 获取当前草稿所有条目

同时，若 `new_word` 已在词库中，调用 `keytao_lookup_by_word(new_word)` 确认其当前编码 `old_code`。

**验证：** 若 `target_code` 不在 `new_word` 的编码链中，告知用户"该编码不是「词」的有效编码位，无法调整到此位置"，终止协议。

**草稿冲突清理（执行 Step 3 前必须先做）：**

检查草稿中是否存在与本次操作冲突的条目——即涉及 `new_word` 或本次受影响编码链（target_code 到最终落点）的任何条目：
- 若草稿里有 `new_word` 的 Create/Delete 条目 → 用 `keytao_batch_remove_draft_items` 删除
- 若草稿里有受影响编码（如 target_code、顺延目标编码）上与本次操作重复的条目 → 一并删除

> 目的：避免草稿中出现重复条目（如 `跑通@pztyo` 已在草稿，顺序调整又要创建 `炮筒@pztyo`，两者都会冲突）。

### Step 2：确定受影响范围

**找出从 `target_code` 到 `new_word` 原编码（`old_code`）之间的所有连续占位词：**

```
chain 中 target_code 的 index = i
若 new_word 已在词库（有 old_code），则 old_code 在 chain 中的 index = j（j > i）
需移动的槽位范围：chain[i], chain[i+1], ..., chain[j-1]
```

若 `new_word` 不在词库，则等效 j = i（无 old_code），只检查从 `target_code` 开始连续被占的槽位：
```
从 chain[i] 开始，依次查 chain[i], chain[i+1], ... 直到遇到空位或 old_code（已释放）
收集所有途中被占的词 → 这些词都需要往后推一格
```

对上述范围中尚未查过的编码，批量调用 `keytao_lookup_by_codes_batch([...])` 查清占用情况。

### Step 3：构建操作列表并一次性提交

> ⚠️ **所有 Delete 和 Create 必须合并为一次 `keytao_batch_add_to_draft` 调用！**
>
> 原因：API 的批内冲突解析（`checkBatchConflictsWithWeight`）只在**同一次调用**的 items 里查找 Delete 来消解 Create 的重码警告。
> 如果分两次调用，第二次 Create 看不到第一次已写入的 Delete，永远会触发 `warnedCount > 0`，进而错误地启动「通用编码自动分配协议」。

**构建顺序（Delete 在前，Create 在后）：**

Delete 部分：
1. 若 `new_word` 已在词库：`{action: Delete, word: new_word, code: old_code}`
2. 对每个被推移的词（从 chain[i] 到 chain[j-1] 的占位词）：`{action: Delete, word: 占位词, code: chain[k]}`

Create 部分：
1. `{action: Create, word: new_word, code: target_code}`
2. 对每个被推移的词：`{action: Create, word: 占位词, code: chain[k+1]}`（往后推一格）

**示例（草稿已有「跑通@pztyo」时）：**
```
// 第一步：删除草稿中的冲突条目
keytao_batch_remove_draft_items(ids=[已有的跑通@pztyo草稿条目ID])

// 第二步：一次性提交所有调整
keytao_batch_add_to_draft(items=[
  {"action": "Delete", "word": "炮筒", "code": "pzty", "type": "Phrase"},
  {"action": "Create", "word": "跑通", "code": "pzty",  "type": "Phrase"},
  {"action": "Create", "word": "炮筒", "code": "pztyo", "type": "Phrase"}
])
```

这样 API 会在同一批次内发现 Delete 炮筒@pzty 解析了 Create 跑通@pzty 的冲突，返回 `warnedCount: 0`，直接进入 Step 4。

### Step 4：回复格式

```
✅ 编码顺序已调整，影响 N 个词条：

📌 顺序变更：
• 跑通 → pzty（原主码位，取代「炮筒」）
• 炮筒 → pztyo（顺延至下一位）

当前草稿（共 N 条）：
...

草稿地址：https://...

发送「提交」以提交该草稿
```

- 每条格式：`词 → 新编码（说明，如"顺延至下一位"或"取代「原词」"）`
- 若只影响 1 个词（目标位为空），按正常加词流程处理，无需展示"顺序变更"段落

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
→ 调用 keytao_submit_batch(confirmed=True)   // 注意：不调其他工具！
→ 成功后回复「批次已提交审核」+ batchUrl，结束
```

### 撤回提审

```
用户：撤回一下
→ 调用 keytao_recall_batch()
→ 批次恢复为草稿状态
```

### 编码顺序调整（用户指定目标编码）

**场景 A：新词抢占已有词的编码（新词不在词库）**

```
[查词流程结束后，AI 展示推荐编码 pztyo，用户说：我要加到 pzty]

→ target_code=pzty, new_word=跑通（不在词库）
→ 调用 keytao_lookup_by_codes_batch(["pzty", "pztyo", "pztyoa"]) 查占用情况
   结果：pzty 有「炮筒」，pztyo 为空
→ 受影响词：炮筒（在 pzty，需移至 pztyo）
→ 一次 batch_add_to_draft，items 顺序：
   [Delete 炮筒@pzty, Create 跑通@pzty, Create 炮筒@pztyo]
→ 汇报：跑通→pzty（取代炮筒），炮筒→pztyo（顺延）
```

**场景 B：词库中已有两词，用户要调换顺序**

```
词库：炮筒@pzty，跑通@pztyo
用户：我要将跑通调到炮筒前面（或：让跑通用 pzty）

→ target_code=pzty, new_word=跑通, old_code=pztyo
→ 受影响范围：chain[pzty] → chain[pztyo]，即炮筒在 pzty 需被推至 pztyo
   （pztyo 因跑通移出而释放，炮筒恰好可去 pztyo）
→ 一次 batch_add_to_draft，items 顺序：
   [Delete 跑通@pztyo, Delete 炮筒@pzty, Create 跑通@pzty, Create 炮筒@pztyo]
→ 汇报：跑通→pzty，炮筒→pztyo（交换顺序）
```

**场景 C：级联推移（多词连续被影响）**

```
词库：词A@pzty，词B@pztyo，pztyoa 为空
用户：我要将跑通加到 pzty

→ 从 pzty 开始连续检查：pzty 有词A，pztyo 有词B，pztyoa 空
→ 受影响词：词A（pzty→pztyo），词B（pztyo→pztyoa）
→ 一次 batch_add_to_draft，items 顺序：
   [Delete 词A@pzty, Delete 词B@pztyo, Create 跑通@pzty, Create 词A@pztyo, Create 词B@pztyoa]
→ 汇报：跑通→pzty（取代词A），词A→pztyo（顺延），词B→pztyoa（顺延）
```

### 编码冲突处理

**`keytao_create_phrase` 或 `keytao_batch_add_to_draft` 遇到重码：**
→ 先判断用户是否明确要求保留重码：
- 是 → 用原编码 + `confirmed=True` 写入，告知用户该词与谁形成重码
- 否（默认）→ 执行「通用编码自动分配协议」，自动找空位编码写入

**`keytao_submit_batch` 返回 `requiresConfirmation: true` 时：**
- 告知用户批次中存在重码，询问是否继续提交
- 用户确认后，**立即重新调用 `keytao_submit_batch(confirmed=True)`**，**绝不改调其他工具**

**Delete 操作无需确认，会直接写入草稿，但成功响应中包含 `notes` 字段**，记录了被删除的词条信息：
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
- **按顺序连续调用工具**，不要在步骤之间停下来等用户确认
- 只有 Create/Change 返回 `requiresConfirmation` 时才暂停询问用户
- Delete 操作**绝不暂停**，直接继续下一步
- 全部完成后一次性汇报结果

### 草稿链接（batchUrl）

所有工具成功返回时都包含 `batchUrl` 字段，指向该草稿批次的网页：
```
https://keytao.vercel.app/batch/{batchId}
```

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

有警告时（`requiresConfirmation: true`），其他步骤已写入草稿，展示格式：
```
✅ 已执行 X 步，当前草稿（共 N 条）：
• ...

⚠️ 第 Y 步有重码：编码 "xxx" 已被 "某词" 占用。
确认继续添加吗？
```

## 回复格式

**提交成功：**
```
✅ 批次已提交审核！共 N 条，等待管理员审核。
批次地址：https://keytao.vercel.app/batch/xxx
```
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
编码 "ushi" 已被 "测试"（权重 0）占用。
添加 "词试" 后将创建重码，两者都会出现在候选框。
确认继续？
```

## 注意事项

1. 只能删除 **Draft** 状态批次中的条目，已提交的无法撤回
2. `code` 必须为纯小写字母，不含空格
3. 草稿批次描述以「键道助手」开头，与用户手动创建的批次区分
