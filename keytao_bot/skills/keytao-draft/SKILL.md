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
| `keytao_list_draft_items` | 查看草稿中所有待审条目 |
| `keytao_remove_draft_item` | 从草稿中移除指定条目 |
| `keytao_submit_batch` | 提交草稿批次进行审核 |

## 工具详解

### keytao_batch_add_to_draft（批量首选）

用户一次提交多个词条时**必须优先使用此工具**（3 个及以上词条）。

**AI 无需传递的参数**（自动注入）：`platform`、`platform_id`

**需要传递的参数：**
- `items` (list): 词条列表，每条包含：
  - `word` (str): 词条内容
  - `code` (str): 键道编码
  - `action` (str, optional): 默认 `Create`
  - `type` (str, optional): 不传则自动推断
  - `remark` (str, optional): 备注

**特性：**
- 遇到硬冲突（词条+编码完全重复、要删除的词不存在等）→ 跳过，记入 `failed`
- 遇到重码警告 → **自动确认写入**，不中断流程
- 草稿中已存在的相同操作 → 跳过，记入 `skipped`
- 最终返回 `successCount`、`failedCount`、`skippedCount`、`failed[]`、`skipped[]`、`draftItems[]`

**返回值示例：**
```json
{
  "success": true,
  "message": "成功写入 27 条，冲突 2 条",
  "batchId": "uuid",
  "successCount": 27,
  "failedCount": 2,
  "skippedCount": 0,
  "failed": [
    {"index": 3, "word": "京豆", "code": "jgddo", "reason": "词条+编码已存在"}
  ],
  "draftItems": [...],
  "draftTotal": 27
}
```

**回复格式：**
```
✅ 批量写入完成！

成功 27 条，冲突 2 条，跳过 0 条。

❌ 冲突条目（未写入）：
• 京豆 jgddo — 词条+编码已存在
• 赠费 zrfwa — ...

当前草稿（共 27 条）：
• 新增 京豆 → jgddo
• 新增 赠费 → zrfwa
...
```

### keytao_create_phrase

**需要传递的参数：**
- `word` (str): 词条内容
- `code` (str): 键道编码（纯字母 a-z）
- `action` (str, optional): `Create`（默认）/ `Change` / `Delete`
- `old_word` (str, optional): Change 操作时的旧词条
- `type` (str, optional): 词条类型，**不传则自动推断**：
  - 1 个汉字 → `Single`；英文字母 → `English`；链接 → `Link`；符号 → `Symbol`；其余 → `Phrase`
- `remark` (str, optional): 备注
- `confirmed` (bool, optional): 确认警告，默认 False

**返回值示例：**
```json
// 成功
{"success": true, "batchId": "uuid", "pullRequestCount": 1, "message": "..."}

// 冲突（不可绕过）
{"success": false, "conflicts": [...], "message": "存在冲突"}

// 警告（可确认后创建）
{"success": false, "warnings": [...], "requiresConfirmation": true, "message": "存在警告"}
```

### keytao_list_draft_items

查看用户当前草稿批次的全部条目。

**AI 无需传递任何参数。**

**返回值示例：**
```json
{
  "success": true,
  "batchId": "uuid",
  "count": 2,
  "items": [
    {"id": 42, "action": "Create", "word": "测试", "code": "ushi", "type": "Phrase", "status": "Pending"},
    {"id": 43, "action": "Create", "word": "若", "code": "ruo", "type": "Single", "status": "Pending"}
  ]
}
```

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

### keytao_submit_batch

提交草稿批次进行管理员审核。**AI 无需传递任何参数。**

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
→ 调用 keytao_list_draft_items()
→ 展示条目列表
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
```

### 重码警告处理

**Create/Change** 操作返回 `requiresConfirmation: true` 时：
- 告知用户重码情况，询问是否继续
- 用户确认后，**立即用相同参数 + `confirmed=True` 重新调用**

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
    "items": [
      {"action": "Delete", "word": "卧推", "code": "wltbv", "type": "Phrase", ...},
      ...
    ]
  }
}
```

**必须在每次操作后展示 `draft_snapshot`**，格式：
```
当前草稿（共 3 条）：
• 删除 卧推 → wltbv
• 删除 我推 → wltb
• 新增 卧推 → wltb
```

有警告时（`requiresConfirmation: true`），其他步骤已写入草稿，展示格式：
```
✅ 已执行 X 步，当前草稿（共 N 条）：
• ...

⚠️ 第 Y 步有重码：编码 "xxx" 已被 "某词" 占用。
确认继续添加吗？
```

## 回复格式

**添加成功：**
```
✅ 词条已加入草稿！
词条：若 | 编码：ruo | 类型：单字
草稿共 3 条，输入「提交」发起审核，或继续添加。
```

**草稿列表：**
```
📋 草稿批次（共 2 条）

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
