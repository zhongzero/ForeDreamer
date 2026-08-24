You are evaluating whether a MemGuide and its available MemTools process retrieved web evidence well enough for downstream forecasting.

Your job is not to write code directly. Your job is to inspect:
1. the current MemGuide,
2. the full source code of the tools available to that guide,
3. a processed rollout summary showing how the guide and tools behaved on one training question.

Decide whether the current guide/tool strategy is already good enough.

If it is good enough:
- return `"should_evolve": false`
- explain why in `"analysis"`
- return an empty string for `"design_requirement"`

If it is not good enough:
- return `"should_evolve": true`
- explain what is insufficient in `"analysis"`
- return a concrete `"design_requirement"` that can be passed into a separate MemGuide/MemTool generator
- the design requirement must describe how to improve the current guide/tool strategy while staying compatible with the same overall MemTool/MemGuide framework
- the design requirement may reuse existing tools and may also request new tools when necessary
- do not return guide JSON or tool code directly

Evaluation criteria:
- whether the processed output captures the most relevant information from the retrieved source
- whether the guide/tool workflow loses important evidence, keeps too much noise, or formats the result poorly
- whether the workflow follows the question-specific requirements well enough
- whether the workflow is robust and general rather than overfitting to one exact example
- whether the current tools are sufficient or need to be extended

Current MemGuide JSON:
{{GUIDE_JSON}}

Current MemTool source code:
{{TOOL_SOURCE_CODE}}

Processed rollout summary:
{{ROLLOUT_SUMMARY}}

Return exactly one JSON object with no markdown fences and no extra text:
{
  "should_evolve": <true_or_false>,
  "analysis": "<short but concrete explanation>",
  "design_requirement": "<empty string if should_evolve is false, otherwise a concrete design requirement for the next guide/tool generation step>"
}
