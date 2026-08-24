You are classifying one newly created MemGuide into an existing set of guide categories.

Each existing category is represented by one representative guide. For each representative guide, you are given:
- `guide_file`
- `guide_name`
- `prompt`
- `tool_names`
- each referenced tool's `TOOL_NAME` and `TOOL_SPEC`

You are also given one new guide candidate with the same fields.

Existing guide category representatives:
{{GUIDE_CATEGORY_REPRESENTATIVES}}

New guide candidate:
{{NEW_GUIDE_CANDIDATE}}

Classification task:
- Decide whether the new guide candidate is highly similar to one existing representative guide category.
- Focus on workflow intent, processing strategy, guide prompt behavior, and the role of the referenced tools.
- Do not focus on filenames or small wording differences alone.
- If two guides share the same high-level state machine, for example both are still "read raw input -> LLM processes -> write final output" two-stage workflows, they should usually be classified into the same category even if tool names, field formatting, headings, or prompt wording differ.
- If the new guide only renames existing reader or writer tools, changes return formatting, adds a few extra fields, or lightly rewrites the prompt while keeping the same stage structure and tool responsibilities, it should still be treated as the existing category.
- Only classify a guide as a new category when it introduces substantial change in stage count, state transitions, intermediate-file usage, tool-type composition, or responsibility boundaries across tools.
- Pay special attention to whether it introduces explicit intermediate artifacts, longer tool chains, or a genuinely different division of roles among tools.
- If the new guide belongs to an existing category, return that representative guide's `guide_file`.
- If the new guide is meaningfully different from every existing category, return `null` so it can become a new category representative.

Required output:
- Return exactly one JSON object and nothing else.
- The JSON object must contain:
  - `matched_representative_guide_file`
  - `analysis`
- `matched_representative_guide_file` must be either one of the provided representative `guide_file` values or `null`.
- `analysis` must be a short explanation.

Valid output example:
```json
{
  "matched_representative_guide_file": "guide_3.json",
  "analysis": "The new guide follows the same extract-then-write workflow pattern and uses tools with the same functional role."
}
```

If no existing category matches, return:
```json
{
  "matched_representative_guide_file": null,
  "analysis": "The new guide introduces a distinct workflow pattern and should become a new category representative."
}
```
