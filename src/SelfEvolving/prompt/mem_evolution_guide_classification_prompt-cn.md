你正在把一个新创建的 MemGuide 归类到现有的 guide 类别中。

每个现有类别都由一个代表 guide 表示。对于每个代表 guide，你会看到：
- `guide_file`
- `guide_name`
- `prompt`
- `tool_names`
- 每个引用工具对应的 `TOOL_NAME` 和 `TOOL_SPEC`

你还会看到一个新增 guide 候选，它也只包含相同字段。

现有 guide 类别代表：
{{GUIDE_CATEGORY_REPRESENTATIVES}}

新增 guide 候选：
{{NEW_GUIDE_CANDIDATE}}

分类任务：
- 判断新增 guide 是否与某个现有代表 guide 所代表的类别高度相似。
- 重点关注工作流目标、处理策略、guide prompt 的行为方式、以及所引用工具承担的角色。
- 不要只根据文件名或少量措辞差异来判断。
- 如果两个 guide 在高层状态机上基本相同，例如都属于“读取原始输入 -> LLM 处理 -> 写最终输出”的两阶段模式，即使工具名、字段格式、段落标题或提示词措辞不同，也应优先判为同一类别。
- 如果新增 guide 只是把已有读工具或写工具换了名字、改了返回文本格式、补了少量字段、或轻微改写 prompt，但阶段结构和工具职责没有明显变化，也应判为已有类别。
- 只有当新增 guide 在阶段数量、状态转移、中间文件使用方式、工具类型组合、或工具职责分工上出现了实质变化时，才应判为新的类别。
- 要特别关注它是否引入了显式中间产物保存/读取、更多阶段的工具链、或与现有代表明显不同的工具角色分工。
- 如果新增 guide 应归入某个已有类别，返回该代表 guide 的 `guide_file`。
- 如果它与所有已有类别都明显不同，返回 `null`，表示它应成为新的类别代表。

输出要求：
- 只能返回一个 JSON 对象，不能有任何额外说明。
- JSON 对象必须包含：
  - `matched_representative_guide_file`
  - `analysis`
- `matched_representative_guide_file` 必须是某个已提供的代表 `guide_file`，或者 `null`。
- `analysis` 必须是简短说明。

合法输出示例：
```json
{
  "matched_representative_guide_file": "guide_3.json",
  "analysis": "这个新增 guide 延续了相同的先提取后写入工作流模式，并且所用工具承担的功能角色一致。"
}
```

如果没有匹配类别，则返回：
```json
{
  "matched_representative_guide_file": null,
  "analysis": "这个新增 guide 引入了明显不同的工作流模式，应成为新的类别代表。"
}
```
