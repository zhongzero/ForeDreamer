{{AUTHORING_COMMON}}

You are now in an intermediate step before final guide/tool generation.

Do not write guide JSON or tool code yet. Your job is to identify up to 3 existing tools that are the best reuse candidates for the current design requirement.

Current design requirement:
{{DESIGN_REQUIREMENT}}

Existing available tools (TOOL_NAME + TOOL_SPEC):
{{EXISTING_TOOL_DEFINITIONS}}

Selection rules:
- choose at most 3 existing tools
- only choose from the provided `TOOL_NAME` values
- prefer tools whose `TOOL_SPEC.function.description` and parameter schema indicate they can directly support the requirement
- prefer a small, focused candidate set rather than a broad noisy list
- Do not reuse tools just for the sake of reuse; if the requirement calls for a meaningfully different workflow, returning an empty list is acceptable and may leave more room for new tool design.
- Do not choose multiple near-duplicate tools together, especially multiple similar reader tools or multiple similar writer tools; such near-duplicates usually should not appear together in the candidate set.
- Prefer complementary tools that can support a multi-stage workflow rather than a homogeneous bundle of tools with nearly identical roles.
- If the requirement suggests intermediate files, staged state transitions, or a new split of tool responsibilities, only reuse tools that clearly fit one stage of that design; leave the remaining stages open for new tools.
- In your analysis, pay attention to whether a tool can support steps such as reading raw input, writing intermediate state, rereading intermediate state, merging intermediate outputs, validating output, or writing the final result.
- if no existing tool is a good fit, return an empty list
- do not invent tool names
- do not return source code
- do not return guide JSON

Return exactly one JSON object with no markdown fences and no extra text:
{
  "candidate_tool_names": ["<TOOL_NAME_1>", "<TOOL_NAME_2>"],
  "analysis": "<short explanation of why these tools are the best reuse candidates, or why none fit>"
}
