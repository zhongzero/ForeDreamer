<div align="center">

# ForeDreamer: A Self-Evolving Dual-Agent Memory Architecture for Future Event Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2608.20920-b31b1b)](https://arxiv.org/abs/2608.20920) [![Project](https://img.shields.io/badge/Project-Page-blue)](https://zhongzero.github.io/ForeDreamer) [![Code](https://img.shields.io/badge/Code-GitHub-orange)](https://github.com/zhongzero/ForeDreamer)

</div>

## 🚀 Overview

Open-web future event prediction requires agents to identify reliable signals in noisy, redundant, incomplete, and sometimes conflicting evidence. ForeDreamer treats this as an **evidence-to-memory transformation** problem: raw search results are converted into structured, question-specific factual memory before the forecasting agent reasons over them.

![ForeDreamer framework overview](https://zhongzero.github.io/ForeDreamer/assets/overview.png)

## 📖 Method

ForeDreamer separates two forms of memory:

- **Factual memory** is the processed evidence state for one forecasting question.
- **Experiential memory** persists across forecasting episodes and guides future search, evidence processing, and prediction.

The framework combines:

- a **main agent** that plans cutoff-aware web searches and produces forecasts;
- a **memory-processing subagent** that follows a MemGuide and executes sandboxed MemTools to transform search results into factual memory;
- **textual experience evolution** that updates an Experience Bank for search planning, evidence integration, and calibration;
- **procedural experience evolution** that updates MemGuides and executable MemTools;
- **Compositional Tool Reuse** and **Diversity-Guided Exploration** for less redundant and more diverse procedural evolution.

This repository focuses on the minimum code path required to evolve and evaluate ForeDreamer. Baseline implementations, ablation runners, large batch schedulers, and historical experiment outputs are not included.

## 🗂️ Repository Structure

```text
.
├── data/
│   ├── FutureX/
│   └── Prophet-arena/
├── scripts/
│   ├── evolve.sh
│   └── test.sh
├── src/
│   ├── SelfEvolving/        # textual and procedural evolution
│   ├── prediction/          # datasets, runners, and metrics
│   ├── DefaultTool/         # search-result processing runtime
│   ├── MemGuide/            # initial evidence-processing guide
│   ├── MemTool/             # initial tools and sandbox runtime
│   ├── experience_bank.py
│   ├── run_eval.py
│   └── summarize_test_results.py
└── requirements.txt
```

The two main entry points are:

- `scripts/evolve.sh`: evolve the Experience Bank, MemGuides, and MemTools, then select the best validated assets.
- `scripts/test.sh`: evaluate the selected assets and produce both per-example predictions and aggregate scores.

## ⚙️ Getting Started

### Requirements

- Linux
- Python 3.10 or newer
- [Bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`) for sandboxed MemTool execution
- an OpenAI-compatible LLM API
- a Tavily or Firecrawl search API

### Environment Setup

Install the system dependency on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y bubblewrap
```

Create an environment and install the Python dependencies:

```bash
conda create -n foredreamer python=3.10 -y
conda activate foredreamer

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export PYTHON_BIN="$(which python)"
```

`pm-rank` is used to compute the Prophet Arena Brier score and average return.

## 🔑 API Configuration

ForeDreamer reads all credentials from environment variables. Do not write API keys into scripts or commit them to the repository.

### LLM API

For OpenAI or another OpenAI-compatible provider:

```bash
export OPENAI_API_KEY="your-llm-api-key"
export OPENAI_MODEL="your-model-name"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

`OPENAI_BASE_URL` defaults to `https://api.openai.com/v1` and may be omitted when using OpenAI directly.

OpenRouter example:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="provider/model-name"
```

### Search API

Tavily is the default search provider:

```bash
export SEARCH_PROVIDER="tavily"
export TAVILY_API_KEY="your-tavily-key"
```

To use Firecrawl:

```bash
export SEARCH_PROVIDER="firecrawl"
export FIRECRAWL_API_KEY="your-firecrawl-key"
```

Both evolution and testing call the LLM and search APIs and may incur usage charges.

## 📦 Data

### FutureX

| File | Purpose |
|---|---|
| `data/FutureX/train-20of208.parquet` | 20 examples used for evolution and validation |
| `data/FutureX/train.parquet` | all 208 resolved examples used as the test input |

Following the paper protocol, the 20 evolution/validation examples are excluded when reporting the held-out score. The remaining evaluation set contains 188 examples.

### Prophet Arena

The repository contains the eight Prophet Arena categories used in the paper:

| Category | Evolution/validation file | Evaluation input |
|---|---|---|
| Climate and Weather | `subset_data_Climate_and_Weather_train_5.csv` | `subset_data_Climate_and_Weather_13.csv` |
| Companies | `subset_data_Companies_train_5.csv` | `subset_data_Companies_27.csv` |
| Economics | `subset_data_Economics_train_5.csv` | `subset_data_Economics_19.csv` |
| Entertainment | `subset_data_Entertainment_train_5.csv` | `subset_data_Entertainment_93.csv` |
| Mentions | `subset_data_Mentions_train_5.csv` | `subset_data_Mentions_26.csv` |
| Other | `subset_data_Other_train_5.csv` | `subset_data_Other_37.csv` |
| Politics | `subset_data_Politics_train_5.csv` | `subset_data_Politics_91.csv` |
| Sports | `subset_data_Sports_train_5.csv` | `subset_data_Sports_200.csv` |

`data/Prophet-arena/subset_data_1200.csv` contains the aggregate 1,200-example set.

## 🧬 Evolve and Evaluate: FutureX

The default evolution command uses the 20-example FutureX evolution/validation set and performs three update iterations as a lower-cost functional run:

```bash
bash scripts/evolve.sh
```

Generated assets are written under `runs/futurex/`. The selected MemGuide and Experience Bank are recorded in `runs/futurex/best_assets.json`.

Test all 208 FutureX examples:

```bash
bash scripts/test.sh
```

The test script uses:

```text
TRAIN_DATA_PATH=data/FutureX/train-20of208.parquet
INPUT_PATH=data/FutureX/train.parquet
```

It predicts all 208 inputs once and reports both:

- the score on all 208 examples;
- the held-out score on the 188 examples remaining after removing the 20 evolution/validation IDs.

To test one or more comma-separated FutureX sample IDs:

```bash
RUN_SPECIFIC="sample-id-1,sample-id-2" bash scripts/test.sh
```

To manually select a MemGuide:

```bash
MEM_GUIDE="guide_2.json" bash scripts/test.sh
```

## 🔬 Paper-scale FutureX Configuration

The paper uses 20 evolution/validation examples, 60 update iterations, four search results per request, and a 1:4 exploration-to-expansion ratio:

```bash
DATASET_TYPE=futurex \
TRAIN_DATA_PATH="./data/FutureX/train-20of208.parquet" \
VAL_DATA_PATH="./data/FutureX/train-20of208.parquet" \
RUN_DIR="./runs/futurex-paper" \
NUM_ITERATIONS=60 \
MEMGUIDE_ROUNDS_PER_CYCLE=10 \
EXPERIENCE_ROUNDS_PER_CYCLE=5 \
EXPLORATION_OVER_EXPANSION=1:4 \
SEARCH_MAX_RESULTS=4 \
PARALLELISM=2 \
bash scripts/evolve.sh
```

Evaluate the resulting assets:

```bash
DATASET_TYPE=futurex \
RUN_DIR="./runs/futurex-paper" \
TRAIN_DATA_PATH="./data/FutureX/train-20of208.parquet" \
INPUT_PATH="./data/FutureX/train.parquet" \
SEARCH_MAX_RESULTS=4 \
bash scripts/test.sh
```

## 🔮 Prophet Arena Example

The following example evolves and evaluates ForeDreamer on the Companies category:

```bash
DATASET_TYPE=prophet_arena \
TRAIN_DATA_PATH="./data/Prophet-arena/subset_data_Companies_train_5.csv" \
VAL_DATA_PATH="./data/Prophet-arena/subset_data_Companies_train_5.csv" \
RUN_DIR="./runs/prophet-companies" \
bash scripts/evolve.sh

DATASET_TYPE=prophet_arena \
TRAIN_DATA_PATH="./data/Prophet-arena/subset_data_Companies_train_5.csv" \
INPUT_PATH="./data/Prophet-arena/subset_data_Companies_27.csv" \
RUN_DIR="./runs/prophet-companies" \
bash scripts/test.sh
```

Use the corresponding files from the data table to run another category.

## 📈 Evaluation Outputs

By default, testing creates:

```text
runs/EXPERIMENT_NAME/predictions.csv
runs/EXPERIMENT_NAME/test_summary.json
```

`predictions.csv` contains per-example model outputs and metrics. `test_summary.json` contains two evaluation scopes:

```json
{
  "all_test": {
    "counts": {},
    "scores": {}
  },
  "train_overlap": {
    "key_field": "id or submission_id",
    "test_rows_in_train": 0,
    "test_rows_without_train": 0
  },
  "test_without_train": {
    "counts": {},
    "scores": {}
  }
}
```

- FutureX overlap is determined with `id` and reports exact-match accuracy.
- Prophet Arena overlap is determined with `submission_id` and reports mean Brier score and mean average return.
- If the test input is already disjoint from the training set, `all_test` and `test_without_train` contain the same examples.

## 🛠️ Configuration

| Environment variable | Default | Description |
|---|---|---|
| `PYTHON_BIN` | `python3.10` | Python executable |
| `DATASET_TYPE` | `futurex` | `futurex` or `prophet_arena` |
| `RUN_DIR` | `runs/$DATASET_TYPE` | Isolated evolution and test output directory |
| `TRAIN_DATA_PATH` | dataset-specific | Evolution data; during testing, IDs from this file are removed from the held-out summary |
| `VAL_DATA_PATH` | `TRAIN_DATA_PATH` | Evolution validation data |
| `INPUT_PATH` | dataset-specific | Complete test/evaluation input |
| `OUTPUT_CSV` | `$RUN_DIR/predictions.csv` | Per-example test output |
| `TEST_SUMMARY_JSON` | `$RUN_DIR/test_summary.json` | Aggregate score summary |
| `NUM_ITERATIONS` | `3` | Number of evolution updates |
| `PARALLELISM` | `1` | Parallel evolution attempts |
| `MEMGUIDE_ROUNDS_PER_CYCLE` | `2` | Procedural updates per cycle |
| `EXPERIENCE_ROUNDS_PER_CYCLE` | `1` | Textual experience updates per cycle |
| `EXPLORATION_OVER_EXPANSION` | `1:1` | Exploration-to-rollout-expansion ratio |
| `SEARCH_MAX_RESULTS` | `1` | Search results per request |
| `MAX_TURNS` | `2` | Maximum main-agent turns |
| `SUBAGENT_MAX_TURNS` | `10` | Maximum memory-processing subagent turns |
| `ENABLE_API_CACHE` | `1` | Cache LLM and search requests under `RUN_DIR/cache` |
| `USE_RAW_CONTEXT` | `1` | Use Tavily `raw_content` when available |
| `RESUME` | `1` | Resume failed or missing test examples from an existing CSV |
| `RUN_SPECIFIC` | unset | Comma-separated event tickers or FutureX sample IDs |
| `MEM_GUIDE` | automatically selected | Override the selected MemGuide |

Show entry-point help:

```bash
bash scripts/evolve.sh --help
bash scripts/test.sh --help
```

## 📁 Run Directory

All generated state is isolated under `RUN_DIR`:

```text
runs/EXPERIMENT_NAME/
├── best_assets.json
├── predictions.csv
├── test_summary.json
├── ExperienceBank/
├── FactualMemory/
├── HistoryEvolution/
├── HistoryRollout/
├── MemGuide/
├── MemTool/
└── cache/
```

The scripts do not modify the initial assets under `src/MemGuide/` or `src/MemTool/`.

## 🖊️ Citation

If you find ForeDreamer useful in your research, please cite:

```bibtex
@article{zhong2026foredreamer,
  title={ForeDreamer: A Self-Evolving Dual-Agent Memory Architecture for Future Event Prediction},
  author={Zhong, Linhao and Du, Zongze and Wu, Linyu and Bo, Yu and Li, Hourong and Jing, Chenchen and Chen, Hao and Xi, Yuling and Shen, Chunhua},
  journal={arXiv preprint arXiv:2608.20920},
  year={2026}
}
```

## 🙏 Acknowledgements

ForeDreamer is evaluated on [Prophet Arena](https://arxiv.org/abs/2510.17638) and [FutureX](https://arxiv.org/abs/2508.11987). We thank the authors of these benchmarks and the open-source projects used in this repository.
