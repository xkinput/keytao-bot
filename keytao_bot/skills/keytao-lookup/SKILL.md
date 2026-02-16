---
name: keytao-lookup
description: 键道输入法查词工具。当用户询问键道输入法编码或词条对应关系时使用。支持按编码查词条、按词条查编码。
version: "1.0.0"
author: xkinput
---

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
- 按编码查询: `GET {KEYTAO_NEXT_URL}/api/phrases/by-code?code={code}&page=1`
- 按词条查询: `GET {KEYTAO_NEXT_URL}/api/phrases/by-word?word={word}&page=1`

**配置变量：**
- `KEYTAO_NEXT_URL`: 默认为 "https://keytao.vercel.app"

**注意事项：**
- 每次查询最多返回 5 条结果（避免信息过载）
- 超时时间设置为 10 秒
- 需要处理网络异常和 API 错误
