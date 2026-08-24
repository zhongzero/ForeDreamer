{{AUTHORING_COMMON}}

You are now in the final generation step.

Current design requirement:
{{DESIGN_REQUIREMENT}}

Candidate reusable existing tools (TOOL_NAME + TOOL_SPEC):
{{REUSABLE_TOOL_DEFINITIONS}}

Candidate reusable existing tool source code:
{{REUSABLE_TOOL_SOURCE_CODE}}

Generation instructions:
- Prefer reusing the provided candidate existing tools whenever they can satisfy the design requirement.
- If an existing tool is sufficient, reference its existing `TOOL_NAME` in the returned MemGuide and do not recreate that tool.
- Only create new MemTool code when the provided existing tools are not enough, are incompatible with the requirement, or have concrete shortcomings.
- It is valid to return only one MemGuide JSON block if no new MemTool code is needed.
- It is also valid to return one MemGuide JSON block plus one or more new MemTool Python code blocks if new tools are required.
- If you decide not to reuse a provided existing tool, make that decision implicitly through the returned workflow. Do not output explanations outside the required code blocks.

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
