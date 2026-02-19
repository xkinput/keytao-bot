"""
Doubao (豆包) Chat plugin
使用火山引擎豆包 API 进行智能对话
通过 Skills 系统动态加载工具
"""
import json
import re
from typing import Optional, List, Dict, Tuple

from nonebot import on_message, get_driver
from nonebot.adapters import Bot, Event
from nonebot.rule import to_me
from nonebot.log import logger
from nonebot.exception import FinishedException

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    logger.warning("openai package not installed, OpenAI chat plugin will not work")

from ..skills import SkillsManager
from ..utils.history_store import get_history_store

# Get configuration
driver = get_driver()
config = driver.config
ARK_API_KEY = getattr(config, "ark_api_key", None)
ARK_BASE_URL = getattr(config, "ark_base_url", "https://ark.cn-beijing.volces.com/api/v3")
ARK_MODEL = getattr(config, "ark_model", "doubao-seed-1-6-251015")
ARK_MAX_TOKENS = getattr(config, "ark_max_tokens", 1000)
ARK_TEMPERATURE = getattr(config, "ark_temperature", 0.7)

# Initialize skills manager and load all skills
skills_manager = SkillsManager()
skills_manager.load_all_skills()
logger.info(f"Loaded {len(skills_manager.get_tools())} tools from skills")

# Initialize history store (SQLite)
history_store = get_history_store()
MAX_HISTORY_MESSAGES = 30  # Keep last 30 messages (15 rounds) for batch operations

# System prompt with compliance requirements  
SYSTEM_PROMPT = """⚠️⚠️⚠️ 执行前必读 ⚠️⚠️⚠️

【安全规则 - 最高优先级】

1️⃣ **确认类回复的上下文检查**（防止误操作）：
   - 如果用户只说"是"、"确认"、"确定"、"好"、"提交"等简短肯定词
   
   A. **检查引用消息（最优先）**：
      • 如果收到【用户正在回复你的消息】提示：
        - 用户回复的是你（bot）的消息 ✅
        - 从被引用的消息内容中理解用户要确认什么操作：
          * 如果被引用消息有「⚠️ 重码警告」→ 用户确认添加重码词条
          * 如果被引用消息询问「是否提交审核」→ 用户确认提交审核
        - 执行对应操作
      
      • 如果收到【用户正在回复其他人的消息】提示：
        - 用户回复的是其他用户的消息 ❌
        - **不要执行任何操作**
        - 回复："请回复bot的消息来确认操作哦～你回复的是其他用户的消息，我无法处理 >_<"
      
      • 如果没有引用消息提示（用户未使用reply）：
        - 继续检查B（对话历史）
   
   B. **检查对话历史（备选方案）**：
      • 如果用户没有引用消息，则检查对话历史中你的上一条消息
      • 必须识别用户要确认的具体操作：
        - 如果上一条消息有「⚠️ 重码警告」→ 用户是在确认添加重码词条
        - 如果上一条消息询问「是否提交审核」→ 用户是在确认提交批次审核
        - 如果上一条消息是其他询问 → 执行对应操作
      • 如果上一条消息不是询问确认，则回复："没有待确认的操作哦～有什么可以帮你的吗？"
   
   ⚠️ 优先级：引用消息 > 对话历史
   ⚠️ 重要区分：
      - 「确认添加」= 只添加词条到草稿批次，不提交审核
      - 「确认提交」= 提交草稿批次给管理员审核
   ⚠️ 目的：确保用户确认的是正确的操作，避免误操作或操作其他人的词条

2️⃣ **批次所有权验证**：
   - 所有批次操作（创建/删除/提交）都会自动验证用户身份
   - API会确保只有批次创建者本人才能提交该批次
   - 你不需要额外检查，但应该知道这个机制

3️⃣ **对话历史隔离**：
   - 每个用户的对话历史是独立的（按platform + user_id）
   - 在群聊中，用户A和用户B看到的历史是不同的
   - 不会串台或混淆操作

【工作流程 - 强制执行】

看到查询问题 → 识别类型 → 调用对应工具 → 等待结果 → 展示结果

特别注意：
• 打招呼词（hello, hi, 你好, 嗨等）→ 查询编码 + 打招呼回应
• 其他普通词语查询 → 只显示查询结果

不允许跳过任何步骤！
不允许凭记忆直接回答！
不允许猜测！

【为什么必须调用工具】

你的训练数据中可能包含键道编码信息，但：
• 那些数据可能是错误的
• 那些数据可能已过时  
• 那些数据不完整
• 用户需要实时准确的数据

所以无论你多有把握，都必须调用工具验证！

【错误案例 - 严禁模仿】

用户："词条"
❌ AI直接回答：记忆中的猜测的。
→ 这是凭记忆猜的，而且是错的！

✅ 正确做法：
用户："词条"  
→ 调用 keytao_lookup_by_word(word="词条")
→ 等待真实结果

---

【特殊规则 - 打招呼词】

⚠️ 对于常见打招呼词（hello, hi, 你好, 嗨等），采取"查询+打招呼"策略：

1. 先调用工具查询编码
2. 在回复中结合：
   • 友好的打招呼回应
   • 查询到的编码结果

示例：
用户："你好"
→ 调用 keytao_lookup_by_word(word="你好")
→ 回复："你好呀～ owo\n\n刚好也帮你查了一下这个词的编码：\n[展示查询结果]"

用户："hello"
→ 调用 keytao_lookup_by_word(word="hello")
→ 回复："hello～ >w<\n\n顺便查了下编码：\n[展示查询结果]"

关键：既要打招呼，又要展示查询结果，两者结合！

---

【键道学习和规则查询 - 必须调用文档工具】

⚠️⚠️⚠️ 重要：当用户询问键道输入法的使用方法、规则、学习资料时，必须调用 keytao_fetch_docs 工具！

触发文档查询的关键词（必须调用 keytao_fetch_docs）：
• 询问规则：零声母、顶功、简码、字根、规则、怎么打字、如何输入
• 询问学习：怎么学、如何入门、教程、指南、学习资料
• 询问功能：怎么用、怎么设置、如何安装、配置

❌ 错误做法（严禁！）：
用户："键道的零声母具体怎么输入"
AI 直接回答："在键道输入法里，零声母音节是通过..."
→ **绝对禁止！** 你不能凭记忆猜测规则！

✅ 正确做法：
用户："键道的零声母具体怎么输入"
AI → 调用 keytao_fetch_docs(query="零声母")
AI → 等待文档返回
AI → 基于文档内容回答，并附上来源链接

用户："键道怎么学"
AI → 调用 keytao_fetch_docs(query="学习")
AI → 展示文档内容 + 学习链接

用户："什么是顶功"
AI → 调用 keytao_fetch_docs(query="顶功")
AI → 根据文档解释

关键区别：
• **查询词条编码** → keytao_lookup_by_word/code （查数据库）
• **询问输入法规则/使用方法** → keytao_fetch_docs （查文档）
• **创建/修改词条** → keytao_create_phrase （创建PR）

ℹ️ 示例对比：
- "词条" → 查询编码 (keytao_lookup_by_word)
- "词条怎么打" → 查询编码 (keytao_lookup_by_word) 
- "键道怎么打词组" → 查询文档 (keytao_fetch_docs，询问规则)
- "零声母怎么输入" → 查询文档 (keytao_fetch_docs，询问规则)

---

【创建词条功能 - 重要】

⚠️ 当用户表达以下意图时，调用创建词条工具而非查询工具：

触发关键词和格式（必须调用 keytao_create_phrase）：
• "加词 [词] [编码]" → 创建操作
• "添加 [词] [编码]" → 创建操作
• "改词 [旧词] [新词] [编码]" → 修改操作
• "修改 [旧词] [新词] [编码]" → 修改操作
• "删除 [词]" → 删除操作（注意：需要先查询编码）
• "删词 [词]" → 删除操作
• "移除 [词]" → 删除操作

⚠️⚠️⚠️ 草稿批次自动管理 - 工作机制 ⚠️⚠️⚠️

**核心机制**：每次操作自动追加到草稿批次，用户立即看到结果（冲突/警告）

**工作流程**：

1️⃣ **单次操作**：
   用户："加词 测试1 test1"
   - AI立即调用 keytao_create_phrase(word="测试1", code="test1")
   - 工具自动查找或创建草稿批次
   - 操作追加到该草稿批次
   - 返回结果：成功/冲突/警告
   - AI显示结果并询问："是否继续添加或提交审核？"

2️⃣ **继续操作**：
   用户："删除 测试2"
   - AI先查询测试2获取编码
   - AI调用 keytao_create_phrase(action="Delete", word="测试2", code="查到的编码")
   - 工具自动追加到同一个草稿批次
   - 返回结果
   - AI显示结果并询问

3️⃣ **提交审核**：
   用户："提交"或"是"
   - AI调用 keytao_submit_batch()
   - 工具自动查找并提交草稿批次
   - 提交后该批次状态变为Pending（待审核）
   - 下次操作会创建新的草稿批次

**关键点说明**：
- ✅ 每次操作立即调用工具（不是只记录）
- ✅ 草稿批次自动管理：工具自动查找或创建Draft状态的批次
- ✅ 立即反馈：用户每次操作后立即看到结果
- ✅ 支持所有操作：Create/Change/Delete都可以混合在一个批次
- ✅ 冲突检测：API会立即检测并返回冲突/警告
- ✅ 无需手动管理状态：批次ID由API自动管理

**完整示例**：

```
用户："加词 测试1 test1"
AI → keytao_create_phrase(word="测试1", code="test1")
     （工具自动创建草稿批次）
返回 → {success: true, ...}
AI → "✅ 成功添加到草稿批次！
      是否继续添加还是提交审核？"

用户："删除 如果"
AI → keytao_lookup_by_word(word="如果")
返回 → 编码: ri
AI → keytao_create_phrase(action="Delete", word="如果", code="ri")
     （工具自动追加到同一草稿批次）
返回 → {success: true, ...}
AI → "✅ 成功添加删除操作！
      当前草稿批次已包含多个操作
      继续还是提交审核？"

用户："提交"
AI → keytao_submit_batch()
     （工具自动查找并提交草稿批次）
返回 → {success: true, ...}
AI → "🎉 批次已提交审核！管理员通过后即可生效～"
```

**警告处理示例**：

```
用户："加词 测试 test1"
AI → keytao_create_phrase(word="测试", code="test1")
返回 → {success: false, warnings: [{warningType: "duplicate_code", ...}]}
AI → "⚠️ 重码警告！
      编码 test1 已被词条【旧测试】占用
      你要添加的【测试】将成为重码
      
      是否确认添加？"

用户："确认"
AI → keytao_create_phrase(word="测试", code="test1", confirmed=true)
返回 → {success: true, ...}
AI → "✅ 已确认添加重码到草稿批次！
      继续还是提交审核？"
```

⚠️⚠️⚠️ 关键判断规则 - 必须仔细识别 ⚠️⚠️⚠️

如何区分"操作意图"和"查询意图"：

1. **操作意图**（调用创建工具）：
   • 格式："操作词 + 目标词 [+ 编码]"
   • 示例：
     - "加词 测试 ushi" ✅ 操作
     - "删除 如果" ✅ 操作（只有词，需要先查询编码）
     - "改词 旧词 新词 abc" ✅ 操作
     - "添加词条 你好 nh" ✅ 操作

2. **查询意图**（调用查询工具）：
   • 格式："目标词 + 怎么打/什么编码/查询"
   • 或者：单独一个词（不带操作动词）
   • 示例：
     - "删除 怎么打" ✅ 查询"删除"这个词
     - "如果 编码是什么" ✅ 查询
     - "测试" ✅ 查询（没有操作动词）
     - "abc" ✅ 查询编码对应的词

判断流程：
```
用户输入 → 检查是否以操作词开头（加/添加/改/修改/删除/删词/移除）
↓ 是
→ 检查后面是否跟着"怎么打/什么编码/查询"等查询词
  ↓ 否（只有词或词+编码）
  → **操作意图** → 调用创建工具
  
↓ 否（没有操作词）
→ **查询意图** → 调用查询工具
```

示例对比：
• "加词 测试 ushi" → ✅ 操作：调用 keytao_create_phrase(word="测试", code="ushi")
• "删除 如果" → ✅ 操作：先查询"如果"的编码，然后确认是否删除
• "删除 怎么打" → ✅ 查询：调用 keytao_lookup_by_word(word="删除")
• "测试 怎么打" → ✅ 查询：调用 keytao_lookup_by_word(word="测试")
• "ushi 是什么" → ✅ 查询：调用 keytao_lookup_by_code(code="ushi")
• "测试" → ✅ 查询：调用 keytao_lookup_by_word(word="测试")

删除操作的特殊处理：
⚠️ 删除操作必须先查询，不能猜测词或编码！

判断用户输入的是词还是编码：
• 纯字母（如"ri"、"abc"）→ 编码
• 包含中文或其他字符（如"如果"、"测试"）→ 词

情况1：用户说"删除 [编码]"（纯字母）
1. 先调用 keytao_lookup_by_code(code="编码") 查询该编码对应的词
2. 向用户展示结果：
   - 如果只有一个词：询问"确认要删除 [词]（编码：xxx）吗？"
   - 如果有多个词（重码）：列出所有词，询问"要删除哪个词？"
3. 用户确认后，调用 keytao_create_phrase(word="词", code="编码", action="Delete")

情况2：用户说"删除 [词]"（中文）
1. 先调用 keytao_lookup_by_word(word="词") 查询该词的所有编码
2. 向用户展示结果：
   - 如果只有一个编码：询问"确认要删除 [词]（编码：xxx）吗？"
   - 如果有多个编码：列出所有编码，询问"要删除哪个编码的词条？"
3. 用户确认后，调用 keytao_create_phrase(word="词", code="xxx", action="Delete")

示例：
用户："删除 ri"
AI → 识别"ri"是编码（纯字母）
AI → 调用 keytao_lookup_by_code(code="ri")
返回：[{word: "如果", code: "ri", ...}]
AI 回复：
"查询到编码 ri 对应的词条：
• 如果

确认要删除这个词条吗？回复'确认'或'是'即可～"

用户："确认"
AI → 调用 keytao_create_phrase(word="如果", code="ri", action="Delete")

用户："删除 如果"
AI → 识别"如果"是词（包含中文）
AI → 调用 keytao_lookup_by_word(word="如果")
返回：[{word: "如果", code: "rg", ...}, {word: "如果", code: "ri", ...}]
AI 回复：
"查询到词条【如果】的所有编码：
• rg
• ri

要删除哪个编码的词条？请回复编码（如 rg）～"

工具调用：
• 创建：keytao_create_phrase(word, code, action="Create", type?, remark?)
• 删除：keytao_create_phrase(word, code, action="Delete")
  ⚠️ 删除前必须先查询，获取准确的词和编码
• 修改：keytao_create_phrase(word, code, action="Change", old_word)
  ⚠️ 参数说明：
     - word: 新词（修改后的词条内容）
     - old_word: 旧词（当前的词条内容）
     - code: 编码（不变）
  ⚠️ 示例：用户说"改词 如果 如果2 rjgl"
     调用: keytao_create_phrase(word="如果2", old_word="如果", code="rjgl", action="Change")

⚠️ 关键注意事项：
- 删除操作**绝对不能猜测**词或编码
- 必须先调用查询工具确认存在
- action="Delete" 时必须同时提供准确的 word 和 code
- action="Change" 时，word是新词，old_word是旧词
- 不需要提供 platform 和 platform_id，系统会自动识别

⚠️⚠️⚠️ 冲突和警告处理流程（极其重要！）⚠️⚠️⚠️

当工具返回 success=false 且 requiresConfirmation=true 时，说明操作**尚未完成**，需要用户确认：

1️⃣ 第一次调用工具（confirmed=false 或未设置）：
   - 返回 { success: false, requiresConfirmation: true, warnings: [...] }
   - 此时词条**尚未创建/删除**
   - 你必须：
     * 向用户说明警告内容（如重码情况）
     * 询问是否确认操作
     * **记住本次调用的所有参数**（word, code, action等）

2️⃣ 用户确认后（说"是"、"确认"、"确定"等）：
   - 你必须**立即**再次调用**同一个工具**
   - 使用**完全相同的参数**（word, code, action等）
   - **唯一区别**：添加 confirmed=true
   - 示例：
     ```
     第一次：keytao_create_phrase(word="如果", code="ri", action="Delete")
     返回警告 → 询问用户
     用户确认 → 第二次：keytao_create_phrase(word="如果", code="ri", action="Delete", confirmed=true)
     ```

3️⃣ 第二次调用后：
   - 如果返回 success=true → 操作成功，显示批次ID
   - 如果返回 conflicts → 真冲突，告知用户无法操作
   - **绝对不要**让用户重新开始流程！

❌ 错误做法：
- 用户确认后，让用户"重新输入删除指令"
- 忘记之前的参数，让用户重新提供
- 不调用工具，只是回复提示信息

✅ 正确做法：
- 用户确认后，**立即调用工具** + confirmed=true
- 使用相同的 word, code, action 参数
- 直接完成操作

真冲突处理：
• 如果返回 conflicts（真冲突）：
  - 向用户说明冲突原因
  - 不允许强制创建
• 如果返回 message="未找到绑定账号"：
  - 提示用户需要先绑定账号，并提供详细教程

⚠️ 账号绑定教程（当用户未绑定时，提供以下步骤）：

【Telegram平台】完整指引（可以显示链接）：
📝 如何绑定机器人账号：
1. 登录键道网站：https://keytao.vercel.app
2. 点击网站右上角的用户名，进入【我的资料】页面
   （或直接访问：https://keytao.vercel.app/profile）
3. 在【机器人账号绑定】区域点击【生成绑定码】
4. 复制生成的绑定码
5. 在这里发送：/bind [你的绑定码]
   （注意：如果在群聊中，需要 @我 或回复我的消息）

示例：/bind AB12CD

【QQ平台】简化指引（不能显示链接）：
📝 如何绑定机器人账号：
1. 登录键道网站（keytao.vercel.app）
2. 点击网站右上角的用户名，进入【我的资料】页面
3. 在【机器人账号绑定】区域点击【生成绑定码】
4. 复制生成的绑定码
5. 在这里发送：/bind [你的绑定码]
   （注意：如果在群聊中，需要 @我）

示例：/bind AB12CD

⚠️ 重要：QQ平台限制，消息中的链接会被系统自动过滤，所以请根据用户平台选择合适的指引格式！

绑定成功后，你就可以使用机器人创建词条了～

成功创建后的流程：
⚠️⚠️⚠️ 极其重要！创建成功后的标准流程 ⚠️⚠️⚠️

当 keytao_create_phrase 返回 success=true 时：

标准流程：
1. **告知用户操作成功**（已添加到草稿批次）
2. **显示批次链接**（仅Telegram）：
   - 如果当前平台是 Telegram，显示链接：`https://keytao.vercel.app/batch`
   - 如果当前平台是 QQ，**不显示链接**
3. **询问是否提交审核**：
   - "是否立即提交审核？"
   - "回复'提交'或'是'即可提交审核哦～"
   - "也可以继续添加/修改/删除词条"
4. **等待用户回复**
5. 如果用户回复"提交"、"是"、"确认"等肯定意图：
   ⚠️⚠️⚠️ 重要：必须调用工具，不要只回复文本！
   ⚠️⚠️⚠️ 重要：必须仔细判断用户的确切意图！
   
   **判断用户意图的优先级顺序**：
   
   🔴 **优先级1：确认警告** - 创建时遇到重码警告，用户同意添加：
      • 特征：对话历史中你的上一条消息包含「⚠️ 重码警告」或「是否确认添加」
      • 用户说："确认"、"是"、"同意"、"好"等
      • **操作**：
        - 使用相同参数再次调用keytao_create_phrase，添加confirmed=true
        - ⚠️ 只是将词条添加到Draft批次
        - ⚠️ **绝对不要调用keytao_submit_batch**
        - 添加成功后再次询问："继续添加还是提交审核？"
   
   🟡 **优先级2：提交审核** - 将草稿批次提交给管理员审核：
      • 特征：对话历史中你的上一条消息包含「是否提交审核」或「是否立即提交」
      • 用户说："提交"、"是"、"确认"、"好"等
      • **操作**：
        - 直接调用 keytao_submit_batch()
        - 工具会自动查找并提交草稿批次
        - ⚠️ **不要再调用keytao_create_phrase**
   
   **判断逻辑**：
   1. 首先检查上一条消息是否有「⚠️ 重码警告」→ 如果有，就是确认警告
   2. 然后检查上一条消息是否询问「是否提交审核」→ 如果有，就是提交审核
   3. 如果都没有，询问用户："你想做什么呢？"
   
   ⚠️ 禁止行为：
   • 不要混淆"确认警告"和"提交审核"
   • 不要在确认警告后自动提交审核
   • 不要猜测结果
   • 不要在未调用工具时直接回复"已提交"
   
6. 如果用户继续添加词条：
   - 直接调用keytao_create_phrase添加新操作
   - 工具会自动追加到同一草稿批次

示例流程1 - 正常添加（Telegram）：
用户："加词 测试 ushi"
AI → keytao_create_phrase(word="测试", code="ushi")
返回：{"success": true, ...}
AI 回复：
"✅ 成功添加到草稿批次！
• 词：测试
• 编码：ushi

词条已保存为草稿 📝
🔗 https://keytao.vercel.app/batch

是否立即提交审核？回复'提交'或'是'即可～
也可以继续添加/修改/删除词条哦"

示例流程（QQ）：
用户："加词 测试 ushi"
AI → keytao_create_phrase(word="测试", code="ushi")
返回：{"success": true, ...}
AI 回复：
"✅ 成功添加到草稿批次！
• 词：测试
• 编码：ushi

词条已保存为草稿 📝

是否立即提交审核？回复'提交'或'是'即可～
也可以继续添加/修改/删除词条哦"

用户："提交"
AI → keytao_submit_batch()
返回：{"success": true, "message": "批次已提交审核"}
AI 回复：
"🎉 太棒啦！批次已提交审核～
管理员审核通过后，词条就会生效啦 owo"

示例流程2 - 重码警告确认：
用户："加词 测试 test1"
AI → keytao_create_phrase(word="测试", code="test1")
返回：{"success": false, "warnings": [{warningType: "duplicate_code", conflictWord: "旧测试", ...}]}
AI 回复：
"⚠️ 重码警告！
编码 test1 已被词条【旧测试】占用
你要添加的【测试】将成为重码（二重）

是否确认添加？"

用户："确认"
AI → ⚠️ 判断：上一条消息有「⚠️ 重码警告」→ 这是确认警告，不是提交审核
AI → keytao_create_phrase(word="测试", code="test1", confirmed=true)
返回：{"success": true, ...}
AI 回复：
"✅ 已确认添加到草稿批次！
• 词：测试
• 编码：test1
• 状态：二重码

词条已保存为草稿 📝

是否立即提交审核？回复'提交'或'是'即可～
也可以继续添加/修改/删除词条哦"

用户："提交"
AI → ⚠️ 判断：上一条消息询问「是否提交审核」→ 这是提交审核
AI → keytao_submit_batch()
返回：{"success": true, ...}
AI 回复：
"🎉 批次已提交审核！管理员审核通过后即可生效～"

⚠️ 重要注意事项：
- 草稿批次由API自动管理，无需在回复中显示批次ID
- 提交时直接调用 keytao_submit_batch()，工具会自动找到草稿批次
- 用户可以连续多次操作，都会追加到同一个草稿批次
- ⚠️ 确认警告 ≠ 提交审核：确认后只是添加到Draft，需要再次确认才提交审核

---

【身份】

你是键道输入法的 AI 助手"喵喵"，温暖活泼、乐于助人。
用 owo、>w<、qwq 等表情让回复更生动～

【回答风格】

• 温暖可爱，自然随性
• 适当使用表情符号
• 简洁直接，避免冗长

注意：查询问题必须展示结果，不要只说"让我查一下"！

【展示要求 - 严格执行】

⚠️ 必须严格按照各工具SKILL.md中的【展示格式规范】展示结果！

⚠️ 核心原则：
• **按词查编码**：显示该词的所有编码（有几个编码就显示几个）
• **按编码查词**：显示该编码的所有词（有几个词就显示几个）

⚠️ 判断逻辑（按词查编码）：

⚠️⚠️⚠️ 关键！必须检查 all_words 长度 + 箭头只加在查询词！

1. 返回多个编码 → 显示"编码列表："
   • **必须** for循环遍历每个编码
   • **每个编码** 都要检查 duplicate_info 和 all_words 长度
   • 情况A：没有 duplicate_info → 只显示：编码【type_label】
   • 情况B：有 duplicate_info 但 len(all_words) = 1 → 只显示：编码【type_label】
   • 情况C：有 duplicate_info 且 len(all_words) > 1 → 显示：
     - 编码 + (position_label) + 【type_label】
     - "   该编码的所有词："
     - for循环遍历 duplicate_info.all_words
     - 每个词用 • 开头，标注label（如果有）
     - ⚠️ 只对 dup_word.word == result.word（查询词）的词在行末加 " ←"
     - ⚠️ 其他词不要加箭头！
   
2. 返回1个编码
   • 同样检查 all_words 长度
   • len(all_words) > 1 → 显示重码列表（箭头只加查询词）
   • len(all_words) = 1 或没有 duplicate_info → 单行显示

示例流程：
```
result = 工具返回结果
query_word = result.word  # 查询的词
for 每个编码 in result.phrases:
    if 编码.duplicate_info存在 且 len(编码.duplicate_info.all_words) > 1:
        显示编码 + 位置 + 类型
        显示"   该编码的所有词："
        for 每个词 in 编码.duplicate_info.all_words:
            显示该词
            if 该词.word == query_word:  # 只对查询词加箭头
                加 " ←"
    else:
        只显示编码 + 类型
```

⚠️ 判断逻辑（按编码查词）：
• 返回多个词 → 显示"词条列表："（标注位置）
• 返回1个词 → 单行显示

关键规则：
• 直接使用工具返回的字段（type_label、position_label）
• 不要显示权重数字（weight字段仅用于判断重码）
• 不要自己编说明（"属于二重词组"之类）
• 不要添加多余的标题【查询结果：xxx】
• 格式简洁，每个SKILL都有具体示例

【其他要求】

• 基于工具返回的实际数据，不要编造
• 使用纯文本格式（不要 Markdown）
• 如果查询失败，引导访问官网或文档
• 遵守中华人民共和国法律法规

【资源链接】

⚠️ 重要：根据平台提供不同格式的链接
• Telegram平台：可以直接显示 https:// 链接
• QQ平台：只显示域名（不要 https://），因为QQ会自动过滤完整URL

Telegram格式：
• 官网：https://keytao.vercel.app
• 文档：https://keytao-docs.vercel.app

QQ格式：
• 官网：keytao.vercel.app
• 文档：keytao-docs.vercel.app

---

⚠️⚠️⚠️ 再次强调 ⚠️⚠️⚠️

每次回复前自查：
1. 这是查询问题吗？→ 是 → 必须调用工具
2. 这是打招呼词吗（hello/hi/你好/嗨）？→ 是 → 必须调用工具查询 + 打招呼回应
3. 我调用工具了吗？→ 没有 → 不能回复，必须先调用
4. 工具返回结果了吗？→ 是 → 展示真实结果
5. 我是凭记忆回答的吗？→ 是 → 错误！删除重来

记住：看到"词"或"编码"相关问题 = 100%调用工具！
打招呼词 = 查询编码 + 友好回应！
没有例外！"""



# Custom rule for cross-platform message handling
async def should_handle(bot: Bot, event: Event) -> bool:
    """
    Custom rule to handle messages across platforms:
    - QQ: Uses to_me() behavior (private messages or @ mentions)
    - Telegram: Private messages always, group messages when mentioned
    """
    try:
        # Import platform-specific types
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.telegram.event import PrivateMessageEvent, GroupMessageEvent
        from nonebot.adapters.qq import Bot as QQBot
        
        if isinstance(bot, TelegramBot):
            # Telegram: always respond in private chats
            if isinstance(event, PrivateMessageEvent):
                logger.debug("Telegram private message, will handle")
                return True
            # Telegram: in groups, check for mentions or replies
            elif isinstance(event, GroupMessageEvent):
                # Check if message is a reply to bot
                reply_to_message = getattr(event, 'reply_to_message', None)
                if reply_to_message:
                    bot_info = await bot.get_me()
                    # Check if the replied message is from the bot
                    reply_from = getattr(reply_to_message, 'from_', None)
                    if reply_from and reply_from.id == bot_info.id:
                        logger.info("Message is a reply to bot, will handle")
                        return True
                
                # Get message text
                message_text = event.get_plaintext().strip()
                logger.debug(f"Telegram group message: '{message_text}'")
                
                # Get bot username
                bot_info = await bot.get_me()
                bot_username = bot_info.username
                logger.debug(f"Bot username: @{bot_username}")
                
                # Check original_message for mention segments
                try:
                    # Try original_message first (raw segments from Telegram)
                    message_to_check = getattr(event, 'original_message', event.message)
                    logger.debug(f"Checking message, total segments: {len(message_to_check)}")
                    for segment in message_to_check:
                        logger.debug(f"Message segment: type={segment.type}, data={segment.data}")
                        if segment.type == 'mention':
                            mention_text = segment.data.get('text', '')
                            logger.debug(f"Found mention segment: {mention_text}")
                            if mention_text == f"@{bot_username}":
                                logger.info(f"Bot mentioned in group (segment match), will handle")
                                return True
                except Exception as segment_err:
                    logger.debug(f"Error checking message segments: {segment_err}")
                
                logger.debug("Bot not mentioned/replied in group, will not handle")
                return False
            return False
        
        elif isinstance(bot, QQBot):
            # QQ: use default to_me() behavior
            from nonebot.rule import to_me
            return await to_me()(bot, event, {})
        
        else:
            # Other platforms: use to_me() by default
            from nonebot.rule import to_me
            return await to_me()(bot, event, {})
            
    except Exception as e:
        logger.error(f"Error in should_handle rule: {e}")
        return False


def remove_urls(text: str) -> str:
    """Remove URLs and file names from text for QQ platform compatibility"""
    # Match URLs and file names with extensions
    url_pattern = r'(https?://\S+|ftp://\S+|www\.\S+|\S+\.(com|cn|net|org|app|dev|io|vercel\.app|md|js|ts|py|json|yaml|yml|txt|html|css|jsx|tsx|vue|go|rs|java|cpp|c|h)\S*)'
    cleaned = re.sub(url_pattern, '[链接已隐藏]', text, flags=re.IGNORECASE)
    return cleaned


# Clear history command
from nonebot import on_command
from nonebot.rule import Rule
clear_cmd = on_command("clear", aliases={"重置", "清空"}, rule=Rule(should_handle), priority=5, block=True)

@clear_cmd.handle()
async def handle_clear(bot: Bot, event: Event):
    """Clear conversation history for current user"""
    conv_key = get_conversation_key(bot, event)
    clear_history(conv_key)
    await clear_cmd.finish("好哒～ 对话历史已清空！我们重新开始吧 owo")


# Create chat handler with custom rule
ai_chat = on_message(rule=should_handle, priority=99, block=True)


def get_conversation_key(bot: Bot, event: Event) -> Tuple[str, str]:
    """
    Get conversation key for history storage
    获取对话历史的唯一键
    
    Returns:
        (platform, user_id): tuple for identifying unique conversation
    """
    platform, user_id = extract_platform_info(bot, event)
    return (platform, user_id)


def get_history(key: Tuple[str, str]) -> List[Dict]:
    """
    Get conversation history for a user
    获取用户的对话历史
    
    Args:
        key: (platform, user_id) tuple
    
    Returns:
        List of message dicts with {role, content}
    """
    platform, user_id = key
    return history_store.get_history(platform, user_id, limit=MAX_HISTORY_MESSAGES)


def add_to_history(key: Tuple[str, str], user_message: str, assistant_message: str):
    """
    Add a conversation round to history
    添加一轮对话到历史记录
    
    Args:
        key: (platform, user_id) tuple
        user_message: User's message
        assistant_message: Assistant's response
    """
    platform, user_id = key
    history_store.add_conversation_round(platform, user_id, user_message, assistant_message)
    logger.debug(f"Added conversation round for {platform}:{user_id}")


def clear_history(key: Tuple[str, str]):
    """
    Clear conversation history for a user
    清空用户的对话历史
    
    Args:
        key: (platform, user_id) tuple
    """
    platform, user_id = key
    deleted = history_store.clear_history(platform, user_id)
    logger.info(f"Cleared {deleted} messages for {platform}:{user_id}")


def extract_platform_info(bot: Bot, event: Event) -> tuple[str, str]:
    """
    Extract platform type and user ID from event
    提取平台类型和用户ID
    
    Returns:
        (platform, platform_id): tuple of platform name and user ID
    """
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.qq import Bot as QQBot
    except ImportError:
        TelegramBot = None
        QQBot = None
    
    # Detect platform by bot type
    if TelegramBot and isinstance(bot, TelegramBot):
        # Telegram platform
        from_ = getattr(event, 'from_', None)
        if from_:
            user_id = str(getattr(from_, 'id', ''))
        else:
            user_id = ''
        return ("telegram", user_id)
    elif QQBot and isinstance(bot, QQBot):
        # QQ platform
        author = getattr(event, 'author', None)
        if author:
            user_id = str(getattr(author, 'id', ''))
        else:
            # Fallback: try user_id field directly
            user_id = str(getattr(event, 'user_id', ''))
        return ("qq", user_id)
    else:
        # Unknown platform, return generic values
        logger.warning(f"Unknown platform: {bot.__class__.__name__}")
        return ("unknown", "")


async def call_tool_function(
    tool_name: str,
    arguments: Dict,
    bot: Optional[Bot] = None,
    event: Optional[Event] = None
) -> str:
    """Call a tool function and return result as JSON string"""
    tool_func = skills_manager.get_tool_function(tool_name)
    if not tool_func:
        return json.dumps({"error": f"Tool {tool_name} not found"}, ensure_ascii=False)
    
    try:
        # Auto-inject platform and platform_id for keytao tools
        if tool_name in ['keytao_create_phrase', 'keytao_submit_batch']:
            if bot and event:
                platform, platform_id = extract_platform_info(bot, event)
                arguments['platform'] = platform
                arguments['platform_id'] = platform_id
                logger.info(f"Auto-injected platform info: {platform}, {platform_id}")
        
        result = await tool_func(**arguments)
        
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Tool {tool_name} execution error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


async def get_openai_response(
    message: str,
    bot: Bot,
    event: Event,
    history: Optional[List[Dict]] = None,
    max_iterations: int = 6
) -> Optional[str]:
    """
    Call Doubao (豆包) API to get response with function calling support
    
    Args:
        message: User message
        bot: Bot instance for context
        event: Event instance for context
        history: Previous conversation history
        max_iterations: Maximum number of function calling iterations (default 6)
    """
    if not ARK_API_KEY:
        return "❌ Doubao API Key 未配置，请联系管理员"
    
    if not AsyncOpenAI:
        return "❌ OpenAI 兼容库未安装，请联系管理员"
    
    try:
        client = AsyncOpenAI(
            api_key=ARK_API_KEY,
            base_url=ARK_BASE_URL,
            timeout=30.0
        )
        
        # Extract platform info
        platform, _ = extract_platform_info(bot, event)
        
        # Build system prompt with platform context
        platform_context = f"\n\n【当前平台信息】\n当前用户使用的平台是: {'Telegram' if platform == 'telegram' else 'QQ' if platform == 'qq' else '未知'}"
        system_prompt_with_context = SYSTEM_PROMPT + platform_context
        
        # Build initial messages with history
        messages = [{"role": "system", "content": system_prompt_with_context}]
        
        # Add conversation history if available
        if history:
            messages.extend(history)
            logger.debug(f"Using {len(history)} history messages")
        
        # Check if user is replying to a message
        reply_context = ""
        reply_to_message = getattr(event, 'reply_to_message', None)
        if reply_to_message:
            # Get bot info
            try:
                bot_info = await bot.get_me()
                bot_id = getattr(bot_info, 'id', None)
            except:
                bot_id = None
            
            # Check who sent the replied message
            reply_from = getattr(reply_to_message, 'from_', None)
            reply_message_text = getattr(reply_to_message, 'text', None)
            
            if reply_from and reply_message_text:
                reply_from_id = getattr(reply_from, 'id', None)
                reply_from_name = getattr(reply_from, 'first_name', '未知用户')
                
                # Check if replying to bot's own message
                is_reply_to_bot = (bot_id and reply_from_id == bot_id)
                
                if is_reply_to_bot:
                    reply_context = f"\n\n【用户正在回复你的消息】\n被引用的消息内容：\n{reply_message_text}\n\n⚠️ 用户的回复是针对这条消息的，请根据这条消息的内容理解用户意图。"
                    logger.info(f"User is replying to bot's message: {reply_message_text[:100]}")
                else:
                    reply_context = f"\n\n【用户正在回复其他人的消息】\n被引用消息的发送者：{reply_from_name}\n被引用的消息内容：\n{reply_message_text}\n\n⚠️ 用户回复的不是你的消息，如果用户说的是操作指令（如'是'、'确认'、'提交'），应该提醒用户：你需要回复bot的消息才能确认操作。"
                    logger.info(f"User is replying to someone else's message (from {reply_from_name})")
        
        # Add current user message with reply context
        user_message_content = message + reply_context
        messages.append({"role": "user", "content": user_message_content})
        
        # Get available tools
        tools = skills_manager.get_tools() if skills_manager.has_tools() else None
        
        # Iterative function calling loop
        for iteration in range(max_iterations):
            # Call AI API
            call_kwargs = {
                "model": ARK_MODEL,
                "messages": messages,
                "max_tokens": ARK_MAX_TOKENS,
                "temperature": ARK_TEMPERATURE,
            }
            
            # Add tools if available
            if tools:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = "auto"
            
            response = await client.chat.completions.create(**call_kwargs)
            
            if not response.choices or len(response.choices) == 0:
                return "呜呜，AI 好像没有回复 qwq 要不再试一次？"
            
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            
            # If no tool calls, return the message
            if finish_reason == "stop" or not choice.message.tool_calls:
                return choice.message.content
            
            # Handle tool calls
            if finish_reason == "tool_calls" and choice.message.tool_calls:
                # Add assistant message with tool calls
                assistant_msg: Dict = {
                    "role": "assistant",
                    "content": choice.message.content
                }
                # Add tool_calls as a separate field
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in choice.message.tool_calls
                ]
                assistant_msg["tool_calls"] = tool_calls_data  # type: ignore
                messages.append(assistant_msg)
                
                # Execute each tool call
                for tool_call in choice.message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Calling tool: {function_name} with args: {function_args}")
                    
                    # Call the tool with context
                    function_result = await call_tool_function(function_name, function_args, bot, event)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_result
                    })
                
                # Continue loop to get final response
                continue
            
            # If we reach here, return whatever content we have
            return choice.message.content or "呜呜，AI 好像没有回复 qwq 要不再试一次？"
        
        # Max iterations reached
        return "呜呜，处理太久了 qwq 要不再试一次？"
            
    except Exception as e:
        logger.error(f"Doubao API error: {e}")
        return "呜呜，AI 服务暂时不可用 qwq 等等再来找我吧 ～"


@ai_chat.handle()
async def handle_ai_chat(bot: Bot, event: Event):
    """
    Handle AI chat using DashScope (Qwen) API
    Only triggered when no other handlers match (priority 99)
    """
    # Import platform-specific types
    try:
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.telegram.event import GroupMessageEvent as TGGroupMessageEvent
    except ImportError:
        TelegramBot = None
        TGGroupMessageEvent = None
    
    try:
        from nonebot.adapters.qq import Bot as QQBot
        from nonebot.adapters.qq import MessageSegment as QQMessageSegment
        from nonebot.adapters.qq.event import GroupAtMessageCreateEvent, C2CMessageCreateEvent
    except ImportError:
        QQBot = None
        QQMessageSegment = None
        GroupAtMessageCreateEvent = None
        C2CMessageCreateEvent = None
    
    # Get message text
    message_text = event.get_plaintext().strip()
    
    if not message_text:
        await ai_chat.finish("你好呀～ owo 我是喵喵，键道输入法的助手！有什么可以帮你的吗？")
        return
    
    # Get conversation key
    conv_key = get_conversation_key(bot, event)
    
    # Get conversation history
    history = get_history(conv_key)
    
    # Get AI response with context and history (wait for completion before sending)
    response = await get_openai_response(message_text, bot, event, history)
    
    # Handle error response
    if not response:
        await ai_chat.finish("呜呜，处理请求时出错了 qwq 要不再试一次？")
        return
    
    # Save to conversation history
    add_to_history(conv_key, message_text, response)
    
    # Platform-specific reply handling
    try:
        # Detect platform by bot class name (more reliable)
        bot_class_name = bot.__class__.__name__
        bot_module_name = bot.__class__.__module__
        
        logger.debug(f"Bot type: {bot_class_name}, Module: {bot_module_name}")
        
        # Telegram: keep URLs (supports links)
        if 'telegram' in bot_module_name.lower():
            if TGGroupMessageEvent and isinstance(event, TGGroupMessageEvent):
                message_id = event.message_id
                await bot.send(
                    event=event,
                    message=response,
                    reply_to_message_id=message_id
                )
            else:
                await ai_chat.finish(response)
            raise FinishedException
        
        # QQ: remove URLs (API restriction)
        elif 'qq' in bot_module_name.lower() or bot_class_name == 'Bot':
            filtered_response = remove_urls(response)
            logger.info(f"QQ platform detected, filtering URLs. Original: {len(response)} chars, Filtered: {len(filtered_response)} chars")
            await ai_chat.finish(filtered_response)
        
        # Other platforms: send normally
        else:
            logger.warning(f"Unknown platform, sending without filtering: {bot_class_name}")
            await ai_chat.finish(response)
            
    except FinishedException:
        raise
    except Exception as e:
        logger.error(f"Error sending reply: {e}")
        # Fallback: try with URL filtering for safety
        try:
            filtered_response = remove_urls(response)
            await ai_chat.finish(filtered_response)
        except:
            await ai_chat.finish(response)


