你正在为 Memory-Evolving-For-Future-Prediction 仓库创建或修改一个 `MemGuide`，并按需要创建或修改一个或多个 `MemTool`，用于 data-process subagent。

你的目标是为单个 `item_dir` 定义一个完整、可执行、无歧义的数据处理工作流。该工作流必须：
- 从 `item_dir/input_filename` 出发
- 通过一个或多个 MemTool 步骤提取、整理、清洗、转换相关信息
- 允许 MemTool 读取源信息，再由 LLM 在对话中直接完成分析、提炼、压缩、改写或生成回答
- 允许在 `item_dir` 下创建中间文件
- 允许 MemTool 直接把 LLM 当前形成的最终回答写入文件
- 最终把所需结果写入 `item_dir/output_filename`

探索与设计偏好：
- 优先考虑具有明确阶段边界的工作流，而不是总是退化成“一个读取工具 + LLM + 一个写入工具”的最短路径。
- 鼓励把复杂处理拆成 3 个或更多有边界的阶段，例如：读取、提取、重组、保存中间结果、再次读取中间结果、压缩、校验、最终写出。
- 鼓励使用不同类型、不同职责的 MemTool，而不是反复生成仅在名字、字段标题或返回文案上略有不同的同类读写工具。
- 鼓励在 `item_dir` 中显式保存和读取中间文件，让工作流状态转移可见、可追踪、可复用。
- 如果需要新工具，优先创建在职责上明显不同的工具，例如：读取原始输入、提取结构化字段、切分内容、写入中间摘要、读取中间摘要、合并多个片段、校验输出、最终写出。
- 不要把“显著不同”理解为仅修改 prompt 措辞、轻微变更字段格式、或把已有 `read_*` / `write_*` 工具换个名字后再次生成。

定义：
- `MemGuide`：data-process subagent 的 JSON 工作流定义，包含 subagent prompt 和允许调用的 MemTool 列表。
- `MemTool`：供 data-process subagent 调用的 Python 工具，每个工具负责当前 `item_dir` 内的一个有边界的处理步骤。
- `item_dir`：单个数据项的工作目录。整个工作流从这个目录中的文件开始，也必须把最终结果写回这个目录。
- `input_filename`：工作流的主输入文件。当前默认示例是 `raw_data.json`。它是一个 UTF-8 编码的 JSON 文件。当前示例结构中至少包含顶层对象 `item`，并且 `item` 至少包含：
  - `title`
  - `url`
  - `content`
  - `published_date`
  - `score`
  其中最重要、最稳定、默认必须正确处理的字段是 `item.content`，应将其视为字符串。当前示例中还可能包含如下顶层元信息字段：
  - `task_id`
  - `run_label`
  - `dataset_name`
  - `run_dir_name`
  - `task_dir_name`
  - `search_turn`
  - `item_rank`
  - `query`
  - `problem_statement`
  - `task_requirements`
  - `search_before`
  - `content_source`
  - `result_count`
- `output_filename`：工作流要求生成的最终输出文件。当前默认示例是 `final_data.txt`。它是一个 UTF-8 编码的纯文本文件，不是 JSON。它的内容应当是从 `input_filename` 中提取、整理、清洗或转换后的最终结果文本。只有当这个文件被写入当前 `item_dir` 后，工作流才算完成。

请严格遵守以下要求：
- 整个工作流必须无歧义且可执行。
- 不要依赖交互输入。
- 不要依赖网络。
- 代码默认使用 ASCII。
- 不要假设沙箱外存在额外文件。

关于输入输出格式，请按以下事实理解并实现：
- 当前默认输入是 `raw_data.json`
- 当前默认输出是 `final_data.txt`
- `raw_data.json` 是结构化 JSON 输入
- `final_data.txt` 是最终纯文本输出
- 如果你的方案依赖 `raw_data.json` 的内容，至少必须正确处理 `item.content`
- 如果你的方案需要使用额外字段，也必须保证这些字段的使用方式与上述 JSON 结构兼容
- 当 `raw_data.json` 中存在 `problem_statement` 和 `task_requirements` 时，应将它们视为当前 item 对应的上游任务问题与要求描述
- 对于 FutureX，这两个字段会对应未来事件问题本身，以及答案要求与最终 boxed answer 约束
- 对于 Prophet Arena，这两个字段会对应要预测的问题，以及最终 JSON 输出要求
- 允许采用“MemTool 读取输入内容 -> LLM 直接生成回答 -> MemTool 将回答写入 `output_filename`”的工作流

运行环境与约束：
- 每个 MemTool 都运行在 `bwrap` 沙箱子进程中。
- 当前 `item_dir` 可读写。
- `/tmp` 是沙箱内私有的可写临时目录，但不属于业务输出目录。
- `src/`、Python runtime、系统目录都是只读的。
- 不允许依赖向 `item_dir` 之外写文件。
- 不允许依赖网络访问。

MemGuide 要求：
- MemGuide 必须是一个 JSON 对象。
- 至少包含以下字段：
  - `guide_name`
  - `prompt`
  - `tool_names`
- `guide_name` 必须非空。
- `prompt` 必须非空。
- `tool_names` 必须是非空列表。
- `tool_names` 中的每个名字都必须对应一个可供 subagent 使用的 MemTool。
- `prompt` 必须明确要求 subagent：
  - 只为当前 `item_dir` 工作
  - 使用给定的一组 MemTool 完成任务
  - 不要编造文件内容
  - 一旦目标输出文件创建成功就停止
- MemGuide 本身不直接执行文件读写；它负责定义 subagent 如何使用 MemTool 完成整个工作流。
- MemGuide 对工作流的描述必须与上述输入输出格式保持一致，不能把 `output_filename` 描述成 JSON，也不能假设 `input_filename` 是任意未定义格式。
- MemGuide 可以采用以下任一模式，或它们的组合：
  - tool 读取输入，另一个 tool 直接处理并写出输出
  - tool 读取输入，LLM 在对话中直接完成推理或生成，随后 tool 把结果写入输出文件
  - tool 读取输入并写中间文件，后续 tool 或 LLM 再基于这些内容完成最终输出
- 当任务允许时，优先考虑包含显式中间状态的多阶段工作流，例如“读取原始输入 -> 提取并写入中间文件 -> 读取中间文件并压缩/重组 -> 校验或定稿 -> 写出最终结果”。

MemTool 要求：
- 每个 MemTool 都必须是一个 Python 文件，对应路径格式为 `src/MemTool/tool_<name>.py`
- 每个 MemTool 都必须定义以下导出：
  - `TOOL_NAME: str`
  - `TOOL_SPEC: dict[str, Any]`
  - `build_runner_kwargs(arguments, config, runtime_context) -> dict[str, Any]`
  - `run_tool(**kwargs) -> str`
  - `__all__ = ["TOOL_NAME", "TOOL_SPEC", "build_runner_kwargs", "run_tool"]`
- 每个 MemTool 的 `TOOL_SPEC` 都必须符合 OpenAI function tool 风格：
  - 顶层必须是 `{"type": "function", "function": {...}}`
  - `function.name` 必须等于 `TOOL_NAME`
  - `function.parameters` 必须使用 JSON Schema object
- 每个 MemTool 的 `run_tool(...)` 都必须：
  - 只返回字符串
  - 对缺失输入文件、格式错误、非法参数抛出清晰的 `ValueError`
  - 只在当前 `item_dir` 内完成业务文件读写
- 单个 MemTool 不一定必须直接读取 `item_dir`、`input_filename`、`output_filename`。
- 但是整个 guide 工作流必须兼容这样一组 runtime context：
  - `item_dir`
  - `input_filename`
  - `output_filename`
- 如果某个 MemTool 直接读取 `input_filename`，应按 UTF-8 JSON 文件处理，并优先围绕 `item.content` 设计逻辑。
- 如果 `problem_statement` 和 `task_requirements` 存在，也允许 MemTool 或 LLM 使用它们来更准确地处理当前 item。
- 如果某个 MemTool 直接写入 `output_filename`，应写出 UTF-8 纯文本，而不是 JSON。
- 如果某个 MemTool 需要特定 runtime 参数，只读取它实际需要的字段并显式校验即可。
- 一个 MemTool 可以只实现整个工作流中的一个阶段，不必单独完成全部任务。
- 允许存在专门的读取型 MemTool，它只负责从 `input_filename` 或中间文件中读取信息并把结果返回给 LLM。
- 允许存在专门的写入型 MemTool，它只负责接收 LLM 当前形成的最终文本，并把该文本写入 `output_filename`。
- 允许存在专门的中间状态工具，例如：
  - 把抽取后的结构化信息写入 `item_dir` 中的中间文件
  - 从中间文件重新读取摘要、片段、候选答案或检查结果
  - 对中间结果进行合并、排序、过滤、校验或重写
- 如果创建多个工具，应尽量让它们承担不同阶段、不同数据形态或不同状态转移职责，而不是复制同一种“读取原文”或“写最终输出”工具。

推荐的最小骨架：

```python
#!/usr/bin/env python3

from pathlib import Path
from typing import Any, Final


TOOL_NAME: Final[str] = "tool_example"
TOOL_SPEC: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Describe what this MemTool does.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


def build_runner_kwargs(arguments: dict[str, Any], config: Any, runtime_context: Any) -> dict[str, Any]:
    del arguments
    del config

    item_dir = str(getattr(runtime_context, "item_dir", "") or "").strip()
    if not item_dir:
        raise ValueError(f"{TOOL_NAME} requires a non-empty runtime item_dir")

    return {
        "item_dir": item_dir,
    }


def run_tool(*, item_dir: str) -> str:
    item_dir_path = Path(item_dir)
    if not item_dir_path.exists():
        raise ValueError(f"{TOOL_NAME} could not find item_dir: {item_dir_path}")

    return f"{TOOL_NAME} completed its step"


__all__ = ["TOOL_NAME", "TOOL_SPEC", "build_runner_kwargs", "run_tool"]
```

如果你的方案需要直接处理主输入输出文件，也可以使用如下风格：

```python
def build_runner_kwargs(arguments: dict[str, Any], config: Any, runtime_context: Any) -> dict[str, Any]:

    item_dir = str(getattr(runtime_context, "item_dir", "") or "").strip()
    input_filename = str(getattr(runtime_context, "input_filename", "") or "").strip()
    output_filename = str(getattr(runtime_context, "output_filename", "") or "").strip()
    if not item_dir:
        raise ValueError(f"{TOOL_NAME} requires a non-empty runtime item_dir")
    if not input_filename:
        raise ValueError(f"{TOOL_NAME} requires a non-empty runtime input_filename")
    if not output_filename:
        raise ValueError(f"{TOOL_NAME} requires a non-empty runtime output_filename")

    return {
        "item_dir": item_dir,
        "input_filename": input_filename,
        "output_filename": output_filename,
    }
```

请特别注意：
- `input_filename = raw_data.json`
- `output_filename = final_data.txt`
- 在最基础示例里，`final_data.txt` 的内容可以直接等于 `raw_data.json["item"]["content"]`
- 在更完整的当前示例里，`raw_data.json` 还可能带有 `problem_statement` 和 `task_requirements`
- 但你的工作流也可以做更合理的提取、清洗、压缩、总结或转换，只要最终输出仍然是纯文本，并且来源可追溯到 `input_filename`
