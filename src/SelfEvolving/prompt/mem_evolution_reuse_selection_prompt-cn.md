{{AUTHORING_COMMON}}

你现在处于最终 guide/tool 生成之前的中间步骤。

此时不要写 guide JSON，也不要写 tool 代码。你的任务是为当前 design requirement 选出最多 3 个最值得复用的已有工具。

当前 design requirement：
{{DESIGN_REQUIREMENT}}

现有可用工具（TOOL_NAME + TOOL_SPEC）：
{{EXISTING_TOOL_DEFINITIONS}}

筛选规则：
- 最多选择 3 个已有工具
- 只能从给定的 `TOOL_NAME` 中选择
- 优先选择那些从 `TOOL_SPEC.function.description` 和参数 schema 看起来就能直接支持当前 requirement 的工具
- 倾向于返回小而精的候选集合，而不是宽泛且嘈杂的列表
- 不要为了复用而复用；如果当前 requirement 需要的是一个明显不同的新工作流，应允许返回空列表，给新工具设计留出空间。
- 不要同时选择多个职责近乎重复的工具，尤其是多个近似的读工具或多个近似的写工具；这类近重复工具通常不应一起进入候选集合。
- 优先选择能够支持多阶段工作流的互补型工具组合，而不是功能同质的工具集合。
- 如果 requirement 明确暗示应引入中间文件、阶段化状态转移、或新的工具职责分工，则只复用那些确实能承担其中某一阶段的工具；其余阶段应留给新工具。
- 在分析时，特别关注工具是否能支持诸如“读取原始输入、写入中间状态、读取中间状态、合并中间结果、校验输出、最终写出”等不同类型的步骤。
- 如果没有合适的已有工具，可以返回空列表
- 不要编造工具名
- 不要返回源代码
- 不要返回 guide JSON

只返回一个 JSON 对象，不要使用 markdown fence，也不要输出任何额外文本：
{
  "candidate_tool_names": ["<TOOL_NAME_1>", "<TOOL_NAME_2>"],
  "analysis": "<简洁说明为什么这些工具最值得复用，或者为什么没有合适候选>"
}
