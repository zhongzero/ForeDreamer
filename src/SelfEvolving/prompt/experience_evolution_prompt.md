You are evaluating whether the main agent's Experience Bank should be updated after one forecasting rollout.

Your job is to inspect:
1. the current Experience Bank (a simplified view that only lists the active experience_id/text pairs),
2. the current best MemGuide validation context,
3. the final run result for the sampled training question,
4. the rollout summary, which includes task metadata, main-agent logs, ground truth, and evaluation feedback.

You must reason about:
- which information in the rollout is directly relevant to the correct answer,
- which parts of the model behavior conflict with the correct answer,
- whether the current information could have been used more effectively to answer the question,
- whether any insight is general and reliable enough to become a reusable experience,
- whether any existing experience is invalid, misleading, redundant, or should be rewritten.

Current Experience Bank:
{{CURRENT_EXPERIENCE_BANK}}

Note: `CURRENT_EXPERIENCE_BANK` only contains the currently active experiences. Each entry only includes `experience_id` and `text`.

Current best guide context:
{{SELECTED_GUIDE_CONTEXT}}

Final run result:
{{RUN_RESULT}}

Rollout summary:
{{ROLLOUT_SUMMARY}}

Note: `ROLLOUT_SUMMARY` only contains the main agent's rollout information. It does not include subagent logs.

Return exactly one JSON object with no markdown fences and no extra text.

Rules:
- Return at most {{MAX_SUGGESTIONS}} suggestions.
- Suggestions must already be ordered from highest priority to lowest priority.
- Use integer priority values where a smaller number means a higher priority.
- `priority = 1` is the highest priority, `priority = 2` is lower, and so on.
- The system will try suggestions in priority order and stop as soon as one suggestion is accepted, so put the most promising suggestion first.
- Valid operations are only `add`, `remove`, and `modify`.
- `remove` and `modify` must reference an existing `target_experience_id`.
- When you use `target_experience_id`, copy it exactly from an `experience_id` field shown in `CURRENT_EXPERIENCE_BANK`.
- Never invent, rename, abbreviate, or paraphrase an existing `target_experience_id`.
- `add` and `modify` must provide a concrete `new_text`.
- Only suggest changes that are genuinely generalizable and likely to improve validation.
- If there are no worthwhile changes, return an empty suggestions list.

Required JSON schema:
{
  "suggestions": [
    {
      "priority": <integer_priority_starting_from_1_where_1_is_highest>,
      "operation": "<add_or_remove_or_modify>",
      "target_experience_id": "<required_for_remove_or_modify_else_empty_string>",
      "new_text": "<required_for_add_or_modify_else_empty_string>",
      "analysis": "<short but concrete reason grounded in the rollout>",
      "generality_assessment": "<why this is general enough or why the old experience is invalid>",
      "expected_benefit": "<how this should improve future validation>"
    }
  ]
}
