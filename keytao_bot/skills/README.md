# Skills 系统

Keytao-bot 的 AI 技能扩展系统，允许 AI 助手通过 Function Calling 自动调用各种工具。

## 📁 目录结构

```
keytao_bot/skills/
├── __init__.py              # Skills Manager
└── keytao-lookup/           # 键道查词 skill
    ├── SKILL.md             # Skill 说明文档
    └── tools.py             # 工具实现
```

## 🎯 工作原理

1. **Skills Manager** 在启动时扫描 `keytao_bot/skills/` 目录
2. 自动加载每个 skill 的 `tools.py` 文件
3. 将工具定义注册到 OpenAI Function Calling
4. AI 根据用户问题自动决定是否调用工具
5. 调用工具后将结果返回给 AI 继续生成回复

## 📝 创建新 Skill

### 1. 创建目录结构

```bash
mkdir -p keytao_bot/skills/your-skill-name
```

### 2. 创建 SKILL.md

```markdown
---
name: your-skill-name
description: 简短描述你的 skill 做什么
version: "1.0.0"
author: your-name
---

# Your Skill Name

详细说明 skill 的功能、使用场景和示例。
```

### 3. 创建 tools.py

```python
"""
Your Skill Tools
"""
from typing import Dict


async def your_tool_function(param: str) -> Dict:
    """
    Tool function description
    
    Args:
        param: Parameter description
        
    Returns:
        dict: Result dictionary
    """
    # Your implementation
    return {
        "success": True,
        "result": "your result"
    }


# Tool definitions for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "your_tool_function",
            "description": "工具描述，AI 会根据这个决定是否调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {
                        "type": "string",
                        "description": "参数描述"
                    }
                },
                "required": ["param"]
            }
        }
    }
]

# Function registry
TOOL_FUNCTIONS = {
    "your_tool_function": your_tool_function
}
```

### 4. 重启机器人

重启后 Skills Manager 会自动加载新的 skill。

## 🔧 Skill 示例：keytao-lookup

### 功能

提供键道输入法的双向查询：
- 按编码查词条（`keytao_lookup_by_code`）
- 按词条查编码（`keytao_lookup_by_word`）

### 使用示例

**用户**: "nau 这个编码对应什么词？"

**AI 行为**:
1. 识别这是查词需求
2. 调用 `keytao_lookup_by_code(code="nau")`
3. 获取结果: `{"success": true, "phrases": [{"word": "你好", ...}]}`
4. 生成友好回复: "nau 对应的词是：你好"

**用户**: "你好 用键道怎么打？"

**AI 行为**:
1. 识别需要查编码
2. 调用 `keytao_lookup_by_word(word="你好")`
3. 获取结果并回复: "你好 的编码是 nau"

## 🧪 测试

运行测试脚本验证 skills 是否正常工作：

```bash
python3 test_skills.py
```

预期输出：
```
============================================================
Testing Skills System
============================================================

1️⃣ Loading skills...
✅ Loaded 2 tools
   1. keytao_lookup_by_code - ...
   2. keytao_lookup_by_word - ...

2️⃣ Testing keytao_lookup_by_code...
   Query result for 'nau':
     • 你好 (nau) [权重: 100]

3️⃣ Testing keytao_lookup_by_word...
   Query result for '你好':
     • 你好 → nau [权重: 100]

============================================================
✅ Skills system test completed
============================================================
```

## 📚 技术细节

### Function Calling 流程

1. AI 收到用户消息
2. 判断是否需要调用工具（基于工具描述）
3. 返回 `finish_reason="tool_calls"` 和工具调用请求
4. Skills Manager 查找并执行对应的函数
5. 将工具执行结果添加到对话历史
6. AI 继续生成回复（可能再次调用工具）
7. 返回最终回复给用户

### 最大迭代次数

默认限制为 3 次迭代，防止无限循环。可在 `openai_chat.py` 中的 `get_openai_response()` 函数修改 `max_iterations` 参数。

### 工具定义格式

遵循 OpenAI Function Calling 标准：
- `type`: 固定为 `"function"`
- `function.name`: 函数名（必须与 TOOL_FUNCTIONS 中的 key 一致）
- `function.description`: 工具描述（AI 据此判断是否调用）
- `function.parameters`: JSON Schema 格式的参数定义

## 🚀 最佳实践

1. **清晰的描述**: 工具描述要准确，让 AI 知道何时调用
2. **简洁的返回**: 工具返回结构化数据，由 AI 生成用户友好的文字
3. **错误处理**: 工具函数要捕获异常，返回包含 `success` 和 `error` 的字典
4. **异步函数**: 所有工具函数都应该是 `async def`
5. **类型提示**: 使用类型注解提高代码可读性

## ⚠️ 注意事项

- 工具名称必须唯一
- 工具描述不要过于宽泛，否则 AI 可能误判
- 工具执行时间不要过长（建议 < 10 秒）
- 返回的数据结构要稳定，便于 AI 理解
