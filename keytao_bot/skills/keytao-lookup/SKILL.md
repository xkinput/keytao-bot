---
name: keytao-lookup
description: 键道输入法查词与编码工具。两类查询：①数据库查询（词库中已有的词）→ keytao_lookup_by_word / keytao_lookup_by_code（无拆分信息）；②规则计算编码（任意词，含拆分）→ keytao_encode。关键规则：用户问拆分/字根/怎么拆/怎么拆分→必须调用keytao_encode，数据库工具没有拆分，禁止替代；加词前→必须先调用keytao_encode生成推荐编码；用户问词库里有什么编码→调用keytao_lookup_by_word；用户问编码对应什么词→调用keytao_lookup_by_code；批量查询优先用batch工具。绝不凭记忆猜测编码，必须调用工具。⚠️展示时严格按照下方【展示格式规范】，不要自己发挥！
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

# keytao_encode：计算编码与字根拆分

此工具按键道规则**实时计算**，适用于所有词（包括词库中没有的词）。与数据库查询不同，它返回完整的逐字拆分信息。

## 触发场景

- "XX 怎么拆" / "XX 的字根是什么" / "XX 是怎么拆分的"
- "XX 怎么打" / "XX 的编码是什么"（只要用户关心拆分，都用此工具）
- 加词流程中（**必须先调用此工具，再添加到草稿**）

⚠️ **关键原则：`keytao_lookup_by_word` / `keytao_lookup_by_code` 均不含拆分信息。  
只要用户问到"怎么拆"、"字根"、"怎么拆分"，无论该词是否在词库，都必须调用 `keytao_encode`，不能用数据库查询代替。**

## keytao_encode 返回结构

```json
{
  "input": "你好",
  "type": "二字词",
  "codes": ["nkhz", "nkhzi", "nkhzia"],
  "altCodes": ["fhz"],
  "chars": [
    {
      "char": "你",
      "pinyin": "ni",
      "phoneticCode": "nk",
      "c1": "人",
      "c2": "丿乙丨",
      "shapeCode": "iuai",
      "fullCode": "nkiuai"
    }
  ]
}
```

字段说明：
- `candidateCodes`：所有可选词条编码（`codes + altCodes` 去重），展示候选和查占用时优先使用此字段
- `recommendedCode`：推荐编码；没有占用信息时通常等于 `codes[0]`
- `codes[0]`：最短规则编码
- `codes[1..]`：逐步加一位形码的选重码
- `altCodes`：飞键备用编码（zh/ch/uang 双键位产生）
- `c1` / `c2`：字根拆分（每个字符是一个字根）
- `shapeCode`：c1+c2 各字根对应的形码字母串
- `phoneticCode`：音码（2字母）

⚠️ 关键规则：词条候选编码只能取 `candidateCodes` / `codes` / `altCodes` / `recommendedCode`，禁止根据 `chars` 里的 `phoneticCode`、`shapeCode`、`fullCode` 自己拼词条编码。

## 展示格式（查询编码/拆分时）

```
「你好」的键道编码（二字词）

推荐编码：nkhz

逐字拆分：
• 你（ni）音码 nk　字根 人｜丿乙丨　形码 iuai
• 好（hao）音码 hz　字根 乙丿｜乙丨　形码 auai

进阶选重：nkhzi · nkhzia
```

若有飞键，追加一行：
```
飞键备用：fhz · ...
```

若某字无拆分数据，形码显示「—」，音码仍正常展示。

## 加词流程（必须 encode 优先）

用户说"帮我加词"、"加词"、"把 XX 加进去"时，**不得直接调用草稿工具**，必须先走以下流程：

### 第一步：调用 keytao_encode

调用 `keytao_encode(word="你好")`，取 `recommendedCode` 作为推荐编码；展示和查占用使用 `candidateCodes`。

### 第二步：发送确认消息

```
「你好」准备加入词库

推荐编码：nkhz（二字词）

逐字拆分：
• 你（ni）音码 nk　字根 人｜丿乙丨
• 好（hao）音码 hz　字根 乙丿｜乙丨

进阶选重：nkhzi · nkhzia

回复 ok / 确认 → 使用推荐编码添加
回复 改码 <编码> → 指定编码，如：改码 nkhzi
回复 取消 → 放弃
```

### 第三步：读取用户回复

| 用户回复 | 处理 |
|----------|------|
| `ok` / `确认` / `是` | 使用 `codes[0]` |
| `改码 <码>` | 使用用户指定的编码 |
| `取消` / `算了` / `不` | 回复"已取消"，结束 |
| 其他 | 提示"请回复 ok、改码 <编码> 或 取消" |

### 第四步：调用草稿工具添加

使用 `keytao-draft` skill 中的 `keytao_create_phrase(word=..., code=...)` 写入草稿。

### 注意事项

- 若 `codes` 为空或含 `?`，提示"该词条暂无法自动编码，请手动指定编码"，让用户用"改码"指定
- 若 API 不可达，提示"编码服务暂时不可用，请稍后再试"
- 编码一律小写展示

---

# 键道输入法查词（数据库查询）

为 AI 助手提供键道输入法的查词能力，支持双向查询：编码→词条 和 词条→编码。

## 使用场景

当用户提出以下问题时，应该调用此 skill：

- "abc 这个编码对应什么词？"
- "nau 怎么打出来的？"（注意：若用户还想了解拆分，改用 keytao_encode）
- "你好 的键道编码是什么？"（查词库）
- "世界 用键道怎么打？"
- "帮我查一下 xxx 的编码"
- "这个编码 xxx 是什么意思"

## 工具定义

### 0. 批量查询规则

- 一次查询多个词的编码时，优先调用 `keytao_lookup_by_words_batch`
- 一次查询多个编码对应的词时，优先调用 `keytao_lookup_by_codes_batch`
- 单个词或单个编码时，调用对应的单查工具
- 批量工具一次最多接收 100 个词或编码

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

### 3. keytao_lookup_by_codes_batch

批量按编码查询词条。

**参数：**

- `codes` (string[], required): 要查询的编码数组，一次最多 100 个

### 4. keytao_lookup_by_words_batch

批量按词条查询编码。

**参数：**

- `words` (string[], required): 要查询的词条数组，一次最多 100 个

### 5. keytao_encode

按键道规则计算编码和字根拆分（非数据库查询）。

**参数：**

- `word` (string, required): 要编码的词条或单字

**必须调用场景：**
- 用户问某词的字根拆分
- 加词流程开始时（必须先调用此工具）

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
