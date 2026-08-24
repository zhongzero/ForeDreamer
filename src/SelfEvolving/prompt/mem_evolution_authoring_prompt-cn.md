{{AUTHORING_COMMON}}

你现在处于最终生成步骤。

当前 design requirement：
{{DESIGN_REQUIREMENT}}

候选可复用已有工具（TOOL_NAME + TOOL_SPEC）：
{{REUSABLE_TOOL_DEFINITIONS}}

候选可复用已有工具源代码：
{{REUSABLE_TOOL_SOURCE_CODE}}

生成要求：
- 如果提供的候选已有工具能够满足 design requirement，应优先复用这些已有工具。
- 如果某个已有工具已经足够，请在返回的 MemGuide 中直接引用它已有的 `TOOL_NAME`，不要重新创建同名工具。
- 只有当提供的已有工具不足、不兼容、或存在明确缺陷时，才创建新的 MemTool 代码。
- 如果不需要新建 MemTool，只返回一个 MemGuide JSON 代码块也是有效的。
- 如果需要新工具，则返回一个 MemGuide JSON 代码块，并在后面追加一个或多个新的 MemTool Python 代码块。
- 如果你判断某个候选已有工具不适合复用，请通过最终返回的工作流隐式体现这个决定，不要在代码块之外输出解释。

返回格式要求：
- 只返回要求的内容，不要有任何其他解释、设计说明、文件名说明或额外 prose。
- 必须先返回且只返回一个 MemGuide 的 JSON 代码块。
- 如果当前任务需要 MemTool，再在后面追加一个或多个完整的 Python 代码块。
- 如果当前任务不需要 MemTool，则只返回 MemGuide 的 JSON 代码块。
- 你的回答去掉前导空白后，开头必须严格是 ```json
- 不允许返回没有 fenced code block 包裹的裸 JSON。
- 不允许在最外层再包额外的 markdown 标题、列表项、说明文字或解释。
- 如果需要返回 MemTool，每个 MemTool 都必须放在各自独立的 ```python fenced code block 中，并且都位于 JSON 代码块之后。
- 如果你的回答没有严格按上述 fenced code block 格式输出，那么它就是无效的。
