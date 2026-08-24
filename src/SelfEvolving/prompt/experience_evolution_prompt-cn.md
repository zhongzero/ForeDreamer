你正在评估：在一次预测 rollout 之后，主 agent 的 Experience Bank 是否应该更新。

你的任务是检查：
1. 当前的 Experience Bank（一个精简视图，只展示当前 active experience 的 `experience_id` 和 `text`），
2. 当前 validation 最优的 MemGuide 上下文，
3. 该训练样本的最终运行结果，
4. rollout summary，其中包含任务元信息、主 agent 日志、GT 和评测反馈。

你必须重点分析：
- rollout 中哪些信息与正确答案直接相关，
- 模型当前哪些行为与正确答案相冲突，
- 基于当前已经拿到的信息，是否存在更有效的信息利用方式从而得到更好的答案，
- 哪些洞见足够通用、足够可靠，值得加入 Experience Bank，
- 哪些现有 experience 无效、误导、冗余，或者应该被改写。

当前 Experience Bank：
{{CURRENT_EXPERIENCE_BANK}}

说明：`CURRENT_EXPERIENCE_BANK` 只包含当前 active experiences，每条只提供 `experience_id` 和 `text`。

当前最优 guide 上下文：
{{SELECTED_GUIDE_CONTEXT}}

最终运行结果：
{{RUN_RESULT}}

Rollout 摘要：
{{ROLLOUT_SUMMARY}}

说明：`ROLLOUT_SUMMARY` 只包含主 agent 的 rollout 信息，不包含 subagent 日志。

请只返回一个 JSON object，不要包含 markdown fence，不要包含额外解释文字。

规则：
- 最多返回 {{MAX_SUGGESTIONS}} 条 suggestion。
- suggestion 必须已经按优先级从高到低排序。
- `priority` 必须是整数，数字越小表示优先级越高。
- `priority = 1` 表示最高优先级，`priority = 2` 表示次高，依此类推。
- 系统会严格按照 priority 从高到低依次尝试 suggestion，并在第一个被接受的 suggestion 处停止，所以最有希望成功的建议必须放在最前面。
- 合法操作只有 `add`、`remove`、`modify`。
- `remove` 和 `modify` 必须引用已有的 `target_experience_id`。
- 使用 `target_experience_id` 时，必须从 `CURRENT_EXPERIENCE_BANK` 中展示的 `experience_id` 字段里原样拷贝。
- 不要自行编造、改写、缩写或意译任何已有的 `target_experience_id`。
- `add` 和 `modify` 必须提供具体的 `new_text`。
- 只有当某条经验真的具有通用性、并且很可能提升 validation 时，才建议修改。
- 如果没有值得执行的修改，返回空的 `suggestions` 列表。

返回 JSON schema：
{
  "suggestions": [
    {
      "priority": <从1开始的整数优先级，其中1表示最高优先级>,
      "operation": "<add_or_remove_or_modify>",
      "target_experience_id": "<remove或modify必填，否则空字符串>",
      "new_text": "<add或modify必填，否则空字符串>",
      "analysis": "<基于rollout的简洁且具体的原因>",
      "generality_assessment": "<为什么该经验足够通用，或者为什么旧经验无效>",
      "expected_benefit": "<预期会如何提升未来validation>"
    }
  ]
}
