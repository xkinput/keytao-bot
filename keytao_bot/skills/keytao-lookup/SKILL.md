---
name: keytao-lookup
description: 键道输入法查词工具。当用户询问词的编码或编码对应的词时必须调用。关键规则：用户问任何词的编码→立即调用keytao_lookup_by_word；用户问任何编码的词→立即调用keytao_lookup_by_code。绝不凭记忆猜测，必须调用工具获取实时数据。词语提取：从用户问题中提取关键词。⚠️展示时必须严格按照下方【展示格式规范】的格式，不要自己发挥！
---
# 展示格式规范

⚠️⚠️⚠️ 重要提醒 ⚠️⚠️⚠️
展示结果时必须：
1. 严格按照下方格式示例展示
2. 使用序号"1. 2. 3."
3. 显示【类型标签】
4. ⚠️ 只有 duplicate_info 存在且 all_words 长度 > 1 时才显示"该编码的所有词："
5. 如果编码只有一个词，即使有 duplicate_info，也只显示编码本身
6. ⚠️ 箭头 ← 只加在当前查询的词后面，其他词不要加！
7. 箭头格式：直接 " ←"（空格+箭头），不要加其他文字
8. 不要自己创造格式！

## 按词查编码 (keytao_lookup_by_word)

⚠️ 核心原则：显示该词的**所有编码**

### 展示流程（务必遵循）

```
1. 看工具返回了几个编码？
   - 1个 → 跳到步骤2
   - 多个 → 显示"编码列表："，继续步骤3

2. 只有1个编码时：
   - 检查这个编码的 duplicate_info 和 all_words 长度
   - 没有 duplicate_info 或 len(all_words) = 1 → 用格式A（单行）
   - 有 duplicate_info 且 len(all_words) > 1 → 用格式B（显示重码）

3. 多个编码时（显示"编码列表："）：
   for 每个编码 in 编码列表:
       显示序号 "1. " "2. " "3. " ...
       
       检查这个编码的 duplicate_info：
         情况1：没有 duplicate_info
           → 显示：编码【type_label】
         
         情况2：有 duplicate_info，但 all_words 长度 = 1（只有当前词）
           → 显示：编码【type_label】（不显示重码列表）
         
         情况3：有 duplicate_info，且 all_words 长度 > 1（真正的重码）
           → 显示：编码 (position_label)【type_label】
           → 换行显示："   该编码的所有词："
           → 遍历 duplicate_info.all_words，每个词用 • 开头
           → ⚠️ 只对 word == result.word（查询词）的词在行末加 " ←"
           → ⚠️ 其他词不要加箭头！
```

### A. 只有1个编码，且无重码

```
词: 找寻
编码: fzxw【词组】
```

### B. 只有1个编码，但该编码有重码

```
词: 执事
编码: fkekiv (三重)【词组】

该编码的所有词：
• 芝士【词组】
• 指事 (二重)【词组】
• 执事 (三重)【词组】 ←
```

⚠️ 注意：箭头只加在查询词"执事"后面，其他词没有箭头！

### C. 有多个编码

**情况C1：多个编码，都无重码**
```
词: 不是

编码列表：
1. bo【声笔笔】
2. bjekvo【词组】
```

**情况C2：多个编码，某些有重码**
```
词: 不实

编码列表：
1. bj【单字】

2. bjekvo (二重)【词组】
   该编码的所有词：
   • 冰激凌【词组】
   • 不实 (二重)【词组】 ←
```

⚠️ 注意：箭头 ← 只加在当前查询词"不实"后面！"冰激凌"没有箭头！

⚠️ 展示C2的代码逻辑：
```python
result = 工具返回结果
query_word = result.word  # 用户查询的词（如"不实"）
phrases = result.phrases  # 编码列表

print(f"词: {query_word}")
print()
print("编码列表：")

for i, phrase in enumerate(phrases, 1):  # 遍历每个编码
    # 检查是否真的有多个词（重码）
    if phrase有duplicate_info 且 len(phrase.duplicate_info.all_words) > 1:
        # 真正的重码，显示详细信息
        print(f"{i}. {phrase.code} ({phrase.duplicate_info.position_label})【{phrase.type_label}】")
        print("   该编码的所有词：")
        
        # 遍历该编码的所有词
        for dup_word in phrase.duplicate_info.all_words:
            if dup_word.label:  # 如果有位置标签
                word_display = f"• {dup_word.word} ({dup_word.label})【{phrase.type_label}】"
            else:
                word_display = f"• {dup_word.word}【{phrase.type_label}】"
            
            # ⚠️ 只对当前查询的词添加箭头！
            if dup_word.word == query_word:  # 比较词是否是查询词
                word_display += " ←"
            
            print(f"   {word_display}")
    else:
        # 没有重码或只有一个词，简单显示
        print(f"{i}. {phrase.code}【{phrase.type_label}】")
    
    print()  # 编码之间空一行
```
    else:
        # 没有重码或只有一个词，简单显示
        print(f"{i}. {phrase.code}【{phrase.type_label}】")
    
    print()  # 编码之间空一行
```

格式说明：
- 只有编码真正有多个词（all_words长度>1）时，才显示"该编码的所有词："
- 如果编码只有一个词，即使有duplicate_info，也只显示编码本身
- 用缩进表示从属关系
- 用 ← 箭头标记当前查询的词

### 判断逻辑

1. 工具返回几个结果？
   - 多个 → 用 C 格式（编码列表）
     * 遍历每个编码，检查 duplicate_info
     * 有 duplicate_info 且 all_words 长度 > 1 → 显示该编码的所有词（C2格式）
     * 没有 duplicate_info 或 all_words 长度 = 1 → 只显示编码（C1格式）
   
2. 工具返回1个结果 → 检查 duplicate_info
   - duplicate_info 存在 且 all_words 长度 > 1 → 用 B 格式（显示重码）
   - duplicate_info 不存在 或 all_words 长度 = 1 → 用 A 格式（单行）

### 数据字段使用

工具返回结构：
```json
{
  "word": "不实",  // 用户查询的词
  "phrases": [...]  // 编码列表
}
```

每个编码包含：
- `code`: 编码字符串
- `word`: 词（就是查询词）
- `type_label`: 类型标签（如"词组"、"单字"）
- `duplicate_info`: 重码信息（如果该编码有重码）
  - `position_label`: 当前词的位置标签（如"二重"、"三重"）
  - `all_words`: 该编码的所有词列表
    - 每个词包含 `word`（词）和 `label`（位置标签）

⚠️ 关键判断：
- 只有当 `len(duplicate_info.all_words) > 1` 时才显示"该编码的所有词："
- 如果 all_words 只有1个，说明该编码只有当前词，不需要显示重码列表

⚠️ 箭头标记规则：
- 使用顶层的 `result.word` 获取查询词
- 在 `all_words` 中，只对 `dup_word.word == result.word` 的词添加 ← 箭头
- 其他词不添加箭头！

展示时：
- 编码后用 `(position_label)` 标注位置（仅当显示重码列表时）
- 编码后用 `【type_label】` 标注类型
- 每个词后标注 `(word.label)` 如果 label 不为空
- 只对查询词在行末添加 " ←"（空格+箭头）

## 按编码查词 (keytao_lookup_by_code)

⚠️ 核心原则：显示该编码的**所有词**

### A. 只有1个词

```
编码: fzxw
词: 找寻【词组】
```

### B. 有多个词（重码）

```
编码: fkekiv

词条列表：
• 芝士【词组】
• 指事 (二重)【词组】
• 执事 (三重)【词组】
```

### 判断逻辑

1. 工具返回几个结果？
   - 只有1个 → 用 A 格式（单行）
   - 多个 → 用 B 格式（词条列表）
2. 显示位置标签：
   - 第1个词：不标注
   - 第2个词：(二重)
   - 第3个词：(三重)
   - 直接使用工具返回的 position_label 字段

# 键道输入法查词

为 AI 助手提供键道输入法的查词能力，支持双向查询：编码→词条 和 词条→编码。

## 使用场景

当用户提出以下问题时，应该调用此 skill：

- "abc 这个编码对应什么词？"
- "nau 怎么打出来的？"
- "你好 的键道编码是什么？"
- "世界 用键道怎么打？"
- "帮我查一下 xxx 的编码"
- "这个编码 xxx 是什么意思"

## 工具定义

### 1. keytao_lookup_by_code

按编码查询词条。

**参数：**

- `code` (string, required): 键道输入法编码，纯字母组合，如 "abc", "nau"

**返回格式：**

```json
{
  "success": true,
  "code": "abc",
  "phrases": [
    {
      "word": "阿爸",
      "code": "abc",
      "weight": 100
    }
  ]
}
```

### 2. keytao_lookup_by_word

按词条查询编码。

**参数：**

- `word` (string, required): 要查询的中文词条，如 "你好", "世界"

**返回格式：**

```json
{
  "success": true,
  "word": "你好",
  "phrases": [
    {
      "word": "你好",
      "code": "nau",
      "weight": 100
    }
  ]
}
```

## 回复建议

当查询成功时，应该友好地展示结果：

**按编码查询示例：**

```
找到编码 "nau" 对应的词条：
• 你好 (nau) [权重: 100]
```

**按词条查询示例：**

```
"你好" 的键道编码是：nau
```

**查询失败时：**

```
抱歉，没有找到相关结果。可能是因为：
- 编码不存在
- 词条不在词库中
- 网络连接问题
```

## 实现细节

**API 端点：**

- 按编码查询: `GET {KEYTAO_API_BASE}/api/phrases/by-code?code={code}&page=1`
- 按词条查询: `GET {KEYTAO_API_BASE}/api/phrases/by-word?word={word}&page=1`

**配置变量：**

- `KEYTAO_API_BASE`: 默认为 "https://keytao.vercel.app"

**注意事项：**

- 每次查询最多返回 5 条结果（避免信息过载）
- 超时时间设置为 10 秒
- 需要处理网络异常和 API 错误
