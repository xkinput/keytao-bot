---
name: keytao-draft
description: 键道输入法草稿批次管理工具。添加词条、查看草稿列表、删除草稿条目、提交审核。关键词：添加、创建、删除、查看、撤销词条，提交审核。使用前需检查用户是否已绑定账号。
---

# 键道草稿批次管理工具

管理用户的键道词条草稿：添加、查看、移除条目，并在就绪时提交审核。

## 使用前提

⚠️ 用户必须先绑定键道平台账号（`/bind`），否则所有工具调用都会返回 404。

## 工具一览

| 工具 | 用途 |
|------|------|
| `keytao_create_phrase` | 添加 / 修改 / 删除词条 |
| `keytao_list_draft_items` | 查看草稿中所有待审条目 |
| `keytao_remove_draft_item` | 从草稿中移除指定条目 |
| `keytao_submit_batch` | 提交草稿批次进行审核 |

## 工具详解

### keytao_create_phrase

添加、修改或删除单个词条。

**AI 无需传递的参数**（自动注入）：`platform`、`platform_id`

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

工具首次返回 `requiresConfirmation: true` → 告知用户重码情况、询问是否继续
用户确认 → Python 层自动以 `confirmed=True` 重新调用，**AI 不需要手动处理**。

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

1. [#42] 新增 "测试" → ushi（词组）
2. [#43] 新增 "若" → ruo（单字）

可以继续添加、删除条目，或输入「提交」发起审核。
```

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
