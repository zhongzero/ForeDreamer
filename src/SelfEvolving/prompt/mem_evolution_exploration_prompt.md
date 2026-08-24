{{AUTHORING_COMMON}}

You are now in the exploration generation step.

Current guide category representatives:
{{GUIDE_CATEGORY_REPRESENTATIVES}}

Exploration objective:
- Create one new MemGuide, and optionally one or more new MemTools, for the data-process subagent.
- The new guide must define a workflow pattern that is clearly different from every existing guide category representative above.
- Do not make only superficial edits such as renaming tools or lightly rewording the prompt.
- The difference should come from workflow design, decomposition strategy, reasoning structure, or tool responsibilities.
- By default, interpret "clearly different" to mean different stage count, different state transitions, different intermediate artifacts, different tool-role decomposition, or a meaningfully different information flow through the workflow.
- If the new design is still just "read raw input -> LLM directly processes -> write final output", it is usually not different enough unless it also introduces a genuinely different tool-role structure or intermediate-state mechanism.
- Prefer exploring multi-stage workflows over shortest-path workflows; when appropriate, explicitly design 3 or more stages.
- Prefer exploring combinations of different tool types such as extraction tools, structured-reorganization tools, intermediate-file writers, intermediate-file readers, fragment-merging tools, validation tools, and final-output writers.
- Strongly favor saving important intermediate results inside `item_dir` and having later tools or the LLM reread them to create clearer stage boundaries.
- Do not fake a new workflow by cloning existing `read_*` / `write_*` tools and changing only small wording details, field headers, or return strings.
- If existing tools are reused, reuse should usually be only a small part of the new workflow; the main novelty should come from new stage structure, tool responsibilities, and intermediate-state design.
- Reusing existing tool names inside the returned MemGuide is allowed only when that truly supports a clearly different guide category.
- If a genuinely new workflow requires new tools, create them.

Examples of encouraged exploration directions:
- "read raw input -> extract into intermediate JSON/TXT -> reread intermediate state and produce a compressed version -> reread the compressed version and write final output"
- "split original content -> summarize chunks -> save merged notes -> reread notes to generate the final result"
- "extract evidence and metadata first -> save candidate conclusions or checklists -> reread that checklist for validation/rewrite -> write final output"
- "separate read / extract / save-intermediate-state / reread-intermediate-state / finalize-write responsibilities across different tools instead of pushing all preprocessing into one reader tool"

Required output format:
- Return only the required content, and nothing else. Do not include explanations, design notes, filename comments, or extra prose.
- You must first return exactly one MemGuide JSON block.
- If the current task requires MemTool code, append one or more complete Python code blocks after the MemGuide JSON block.
- If the current task does not require MemTool code, return only the MemGuide JSON block.
- The first non-whitespace characters of your answer must be exactly ```json
- Do not return raw JSON without a fenced code block.
- Do not wrap the whole answer in any outer markdown section, list item, heading, or explanation.
- If you need to return MemTool code, each MemTool must appear in its own separate ```python fenced code block after the JSON block.
- If your answer is not formatted as fenced code blocks exactly as required below, it is invalid.
