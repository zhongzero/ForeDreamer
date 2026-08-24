你要评估一个 MemGuide 及其可用 MemTool，判断它们对检索到的网页证据进行处理的策略是否足够好，能够支持下游预测任务。

你的任务不是直接写代码。你需要检查：
1. 当前的 MemGuide
2. 当前 guide 可用的所有 MemTool 的完整源代码
3. 一个处理后的 rollout 摘要，用来展示该 guide 和这些 tool 在一条训练问题上的实际表现

你需要判断当前的 guide/tool 策略是否已经足够好。

如果已经足够好：
- 返回 `"should_evolve": false`
- 在 `"analysis"` 中解释原因
- `"design_requirement"` 返回空字符串

如果还不够好：
- 返回 `"should_evolve": true`
- 在 `"analysis"` 中说明当前策略哪里不足
- 在 `"design_requirement"` 中返回一个具体的设计要求，这个设计要求会被传给另一个 MemGuide/MemTool 生成器
- 这个设计要求必须描述如何在当前 guide/tool 策略的基础上进行优化，同时保持与现有 MemTool/MemGuide 框架兼容
- 设计要求可以复用已有 tool，也可以在必要时要求创建新 tool
- 不要直接返回 guide JSON 或 tool 代码

评估标准：
- 处理后的输出是否提取并保留了检索结果中最相关的信息
- guide/tool 工作流是否丢失了重要证据、保留了过多噪声、或输出格式不佳
- 工作流是否足够符合当前问题的要求
- 工作流是否具有一般性和鲁棒性，而不是只对当前样例过拟合
- 当前工具集合是否足够，或者是否需要扩展

当前 MemGuide JSON：
{{GUIDE_JSON}}

当前 MemTool 源代码：
{{TOOL_SOURCE_CODE}}

处理后的 rollout 摘要：
{{ROLLOUT_SUMMARY}}

只返回一个 JSON 对象，不要使用 markdown fence，也不要输出任何额外文本：
{
  "should_evolve": <true_or_false>,
  "analysis": "<简洁但具体的解释>",
  "design_requirement": "<如果 should_evolve 为 false 则为空字符串；如果为 true，则返回一个具体的设计要求，用于下一步 guide/tool 生成>"
}
