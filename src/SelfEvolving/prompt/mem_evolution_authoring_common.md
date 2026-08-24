You are working inside the Memory-Evolving-For-Future-Prediction repository to create or update one `MemGuide`, and optionally create or update one or more `MemTool` files, for the data-process subagent.

Your goal is to define a complete, executable, unambiguous data-processing workflow for a single `item_dir`. The workflow must:
- start from `item_dir/input_filename`
- use one or more MemTool steps to extract, organize, clean, or transform relevant information
- allow a MemTool to read source information and let the LLM directly analyze, compress, rewrite, or generate the answer in the conversation
- optionally create intermediate files under `item_dir`
- optionally let a MemTool directly write the LLM's final answer into a file
- finally write the required result to `item_dir/output_filename`

Exploration and design preferences:
- Prefer workflows with explicit stage boundaries instead of repeatedly collapsing to the shortest possible "one reader tool + LLM + one writer tool" pattern.
- Favor decomposing complex processing into 3 or more bounded stages when appropriate, for example: read, extract, reorganize, save intermediate state, reread intermediate state, compress, validate, and finally write.
- Favor MemTools with clearly different responsibilities rather than repeatedly generating near-duplicate reader or writer tools that differ only in names, field labels, or return wording.
- Favor explicit saving and rereading of intermediate files inside `item_dir` so workflow state transitions are visible, traceable, and reusable.
- When new tools are needed, prefer tools with clearly distinct roles such as: raw-input reader, structured-field extractor, content splitter, intermediate-summary writer, intermediate-summary reader, fragment merger, output validator, or final writer.
- Do not treat small prompt rewrites, slight field-format changes, or renamed variants of existing `read_*` / `write_*` tools as meaningful novelty.

Definitions:
- `MemGuide`: a JSON workflow definition for the data-process subagent. It contains the subagent prompt and the list of MemTools the subagent is allowed to call.
- `MemTool`: a Python tool callable by the data-process subagent. Each MemTool performs one bounded processing step inside the current `item_dir`.
- `item_dir`: the working directory for one data item. The workflow starts from files inside this directory and must also write its final result inside this directory.
- `input_filename`: the main input file for the workflow. The current default example is `raw_data.json`. It is a UTF-8 encoded JSON file. In the current example structure, it contains at least a top-level `item` object, and `item` contains at least:
  - `title`
  - `url`
  - `content`
  - `published_date`
  - `score`
  The most important, most stable, and default-required field is `item.content`, which should be treated as a string. The current example may also contain top-level metadata fields such as:
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
- `output_filename`: the required final output file for the workflow. The current default example is `final_data.txt`. It is a UTF-8 encoded plain-text file, not JSON. Its contents should be the final extracted, transformed, cleaned, or organized text derived from `input_filename`. The workflow is only complete when this file is written under the current `item_dir`.

Follow these requirements strictly:
- Always keep the workflow unambiguous and executable.
- Do not depend on interactive input.
- Do not depend on network access.
- Use ASCII by default in code.
- Do not assume extra files exist outside the sandbox.

Interpret the input and output formats according to these facts:
- the current default input is `raw_data.json`
- the current default output is `final_data.txt`
- `raw_data.json` is structured JSON input
- `final_data.txt` is final plain-text output
- if your design depends on `raw_data.json`, it must at minimum handle `item.content` correctly
- if your design uses additional fields, that usage must remain compatible with the JSON structure described above
- when `raw_data.json` contains `problem_statement` and `task_requirements`, treat them as the upstream task problem and requirement description for the current item
- for FutureX, these fields correspond to the event question plus answer requirements and final boxed-answer constraints
- for Prophet Arena, these fields correspond to the prediction question plus the final JSON output requirements
- a valid workflow may be: a MemTool reads the input, the LLM directly produces the answer in the conversation, and another MemTool writes that answer to `output_filename`

Runtime environment and constraints:
- Every MemTool runs inside a `bwrap` sandbox subprocess.
- The current `item_dir` is readable and writable.
- `/tmp` is a private writable temporary directory inside the sandbox, but it is not part of the business output directory.
- `src/`, the Python runtime, and system directories are read-only.
- Do not depend on writing outside `item_dir`.
- Do not depend on network access.

MemGuide requirements:
- A MemGuide must be a JSON object.
- It must contain at least:
  - `guide_name`
  - `prompt`
  - `tool_names`
- `guide_name` must be non-empty.
- `prompt` must be non-empty.
- `tool_names` must be a non-empty list.
- Every name in `tool_names` must correspond to a MemTool available to the subagent.
- The `prompt` must explicitly require that the subagent:
  - works only for the current `item_dir`
  - uses the provided MemTool set to finish the task
  - does not invent file contents
  - stops once the target output file has been created
- A MemGuide does not perform file I/O itself. It defines how the subagent should use the MemTools to complete the workflow.
- The MemGuide description of the workflow must remain consistent with the input/output formats above. Do not describe `output_filename` as JSON, and do not assume `input_filename` has an arbitrary undefined format.
- A MemGuide may use any of the following patterns, or a combination of them:
  - a tool reads input and another tool directly processes and writes output
  - a tool reads input, the LLM directly reasons or generates the answer in the conversation, and a tool then writes that answer to the output file
  - a tool reads input and writes intermediate files, and later tools or the LLM use those intermediate results to produce the final output
- When the task allows it, prefer multi-stage workflows with explicit intermediate state, such as "read raw input -> extract and write intermediate state -> reread intermediate state and compress/reorganize -> validate or finalize -> write final output".

MemTool requirements:
- Every MemTool must be a Python file corresponding to path format `src/MemTool/tool_<name>.py`
- Every MemTool must define these exports:
  - `TOOL_NAME: str`
  - `TOOL_SPEC: dict[str, Any]`
  - `build_runner_kwargs(arguments, config, runtime_context) -> dict[str, Any]`
  - `run_tool(**kwargs) -> str`
  - `__all__ = ["TOOL_NAME", "TOOL_SPEC", "build_runner_kwargs", "run_tool"]`
- Every MemTool `TOOL_SPEC` must follow OpenAI function-tool style:
  - the top level must be `{"type": "function", "function": {...}}`
  - `function.name` must equal `TOOL_NAME`
  - `function.parameters` must use a JSON Schema object
- Every MemTool `run_tool(...)` must:
  - return a string only
  - raise clear `ValueError` messages for missing input files, invalid formats, or invalid arguments
  - perform business file reads and writes only inside the current `item_dir`
- An individual MemTool does not have to read `item_dir`, `input_filename`, and `output_filename` directly.
- However, the overall guide workflow must be compatible with a runtime context that provides:
  - `item_dir`
  - `input_filename`
  - `output_filename`
- If a MemTool directly reads `input_filename`, it should treat it as a UTF-8 JSON file and should prioritize logic around `item.content`.
- If `problem_statement` and `task_requirements` exist, the MemTool or the LLM may use them to process the current item more accurately.
- If a MemTool directly writes `output_filename`, it should write UTF-8 plain text, not JSON.
- If a MemTool needs specific runtime parameters, read only the fields it actually needs and validate them explicitly.
- A MemTool may implement only one stage of the workflow. It does not need to complete the whole workflow by itself.
- Reader-style MemTools are allowed: they may only read `input_filename` or intermediate files and return the relevant information to the LLM.
- Writer-style MemTools are allowed: they may only take the LLM's current final text and write it to `output_filename`.
- Intermediate-state MemTools are also allowed, for example tools that:
  - write extracted structured data into intermediate files inside `item_dir`
  - reread summaries, fragments, candidate answers, or checklists from intermediate files
  - merge, sort, filter, validate, or rewrite intermediate results
- If you create multiple tools, try to give them meaningfully different roles across stages, data shapes, or state transitions instead of duplicating the same "read original input" or "write final output" function.

Recommended minimal skeleton:

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

If your design needs direct access to the main input or output files, you may also use this style:

```python
def build_runner_kwargs(arguments: dict[str, Any], config: Any, runtime_context: Any) -> dict[str, Any]:
    del arguments
    del config

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

Special attention:
- `input_filename = raw_data.json`
- `output_filename = final_data.txt`
- in the most basic example, `final_data.txt` may simply equal `raw_data.json["item"]["content"]`
- in richer examples, `raw_data.json` may also include `problem_statement` and `task_requirements`
- your workflow may still perform better extraction, cleaning, compression, summarization, or transformation, as long as the final output remains plain text and is traceable to `input_filename`
