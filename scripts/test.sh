#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "${1:-}" == "--help" ]]; then
  sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

# Usage:
#   bash scripts/test.sh
# Run scripts/evolve.sh first. Configure with environment variables; see README.md.

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
DATASET_TYPE="${DATASET_TYPE:-futurex}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/$DATASET_TYPE}"
SEARCH_PROVIDER="${SEARCH_PROVIDER:-tavily}"
SEARCH_MAX_RESULTS="${SEARCH_MAX_RESULTS:-1}"
SEARCH_MAX_CHARS_PER_RESULT="${SEARCH_MAX_CHARS_PER_RESULT:-30000}"
SEARCH_MAX_TOTAL_CHARS="${SEARCH_MAX_TOTAL_CHARS:-30000}"
MAX_TURNS="${MAX_TURNS:-2}"
SUBAGENT_MAX_TURNS="${SUBAGENT_MAX_TURNS:-10}"
ENABLE_API_CACHE="${ENABLE_API_CACHE:-1}"
USE_RAW_CONTEXT="${USE_RAW_CONTEXT:-1}"
RESUME="${RESUME:-1}"

case "$DATASET_TYPE" in
  futurex)
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$REPO_DIR/data/FutureX/train-20of208.parquet}"
    INPUT_PATH="${INPUT_PATH:-$REPO_DIR/data/FutureX/train.parquet}"
    ;;
  prophet_arena)
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$REPO_DIR/data/Prophet-arena/subset_data_Companies_train_5.csv}"
    INPUT_PATH="${INPUT_PATH:-$REPO_DIR/data/Prophet-arena/subset_data_Companies_27.csv}"
    ;;
  *)
    echo "DATASET_TYPE must be futurex or prophet_arena" >&2
    exit 2
    ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "ForeDreamer requires Python 3.10 or newer" >&2
  exit 2
}
if [[ -z "${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}" || -z "${OPENAI_MODEL:-}" ]]; then
  echo "Set OPENAI_API_KEY/OPENROUTER_API_KEY and OPENAI_MODEL" >&2
  exit 2
fi
case "$SEARCH_PROVIDER" in
  tavily)
    [[ -n "${TAVILY_API_KEY:-}" ]] || { echo "Set TAVILY_API_KEY" >&2; exit 2; }
    ;;
  firecrawl)
    [[ -n "${FIRECRAWL_API_KEY:-}" ]] || { echo "Set FIRECRAWL_API_KEY" >&2; exit 2; }
    ;;
  *)
    echo "SEARCH_PROVIDER must be tavily or firecrawl" >&2
    exit 2
    ;;
esac
if [[ ! -x "$(command -v bwrap 2>/dev/null || true)" ]]; then
  echo "bubblewrap is required; install the bwrap command first" >&2
  exit 2
fi
[[ -f "$INPUT_PATH" ]] || { echo "Test data not found: $INPUT_PATH" >&2; exit 2; }
[[ -f "$TRAIN_DATA_PATH" ]] || { echo "Training data not found: $TRAIN_DATA_PATH" >&2; exit 2; }

FACTUAL_MEMORY_DIR="$RUN_DIR/FactualMemory"
HISTORY_EVOLUTION_DIR="$RUN_DIR/HistoryEvolution"
HISTORY_ROLLOUT_DIR="$RUN_DIR/HistoryRollout"
MEMGUIDE_DIR="$RUN_DIR/MemGuide"
MEMTOOL_DIR="$RUN_DIR/MemTool"
EXPERIENCE_BANK_DIR="$RUN_DIR/ExperienceBank"
CACHE_DIR="$RUN_DIR/cache"
OUTPUT_CSV="${OUTPUT_CSV:-$RUN_DIR/predictions.csv}"
TEST_SUMMARY_JSON="${TEST_SUMMARY_JSON:-$RUN_DIR/test_summary.json}"

[[ -f "$MEMGUIDE_DIR/evolving_tree.json" ]] || {
  echo "No evolved assets found under $RUN_DIR; run scripts/evolve.sh first" >&2
  exit 2
}

cd "$REPO_DIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

if [[ -z "${MEM_GUIDE:-}" ]]; then
  MEM_GUIDE="$("$PYTHON_BIN" src/select_best_assets.py \
    --dataset_type "$DATASET_TYPE" \
    --memguide_dir "$MEMGUIDE_DIR" \
    --experience_bank_dir "$EXPERIENCE_BANK_DIR" \
    --field guide_file)"
fi

OPTIONAL_FLAGS=()
[[ "$ENABLE_API_CACHE" == "1" ]] && OPTIONAL_FLAGS+=(--enable_api_cache --disable_main_agent_final_answer_cache)
[[ "$USE_RAW_CONTEXT" == "1" ]] && OPTIONAL_FLAGS+=(--use_tavilty_raw_context)
[[ "$RESUME" == "1" ]] && OPTIONAL_FLAGS+=(--resume_from_output_csv)
[[ "$DATASET_TYPE" == "prophet_arena" ]] && OPTIONAL_FLAGS+=(--use_market_data_in_prophet_arena)
if [[ -n "${RUN_SPECIFIC:-}" ]]; then
  RUN_FLAGS=(--run_specific "$RUN_SPECIFIC")
else
  RUN_FLAGS=(--run_all)
fi

echo "Starting ForeDreamer test"
echo "dataset=$DATASET_TYPE guide=$MEM_GUIDE train=$TRAIN_DATA_PATH input=$INPUT_PATH output=$OUTPUT_CSV"
"$PYTHON_BIN" src/run_eval.py \
  --input_csv "$INPUT_PATH" \
  --output_csv "$OUTPUT_CSV" \
  --dataset_type "$DATASET_TYPE" \
  --strategy web_search_loop \
  --max_turns "$MAX_TURNS" \
  --subagent_max_turns "$SUBAGENT_MAX_TURNS" \
  --search_provider "$SEARCH_PROVIDER" \
  --search_max_results "$SEARCH_MAX_RESULTS" \
  --search_max_chars_per_result "$SEARCH_MAX_CHARS_PER_RESULT" \
  --search_max_total_chars "$SEARCH_MAX_TOTAL_CHARS" \
  --mem_guide "$MEM_GUIDE" \
  --experience current.json \
  --factual_memory_dir "$FACTUAL_MEMORY_DIR" \
  --history_evolution_dir "$HISTORY_EVOLUTION_DIR" \
  --history_rollout_dir "$HISTORY_ROLLOUT_DIR" \
  --memguide_dir "$MEMGUIDE_DIR" \
  --memtool_dir "$MEMTOOL_DIR" \
  --experience_bank_dir "$EXPERIENCE_BANK_DIR" \
  --cache_dir "$CACHE_DIR" \
  "${RUN_FLAGS[@]}" \
  "${OPTIONAL_FLAGS[@]}"

"$PYTHON_BIN" src/summarize_test_results.py \
  --dataset_type "$DATASET_TYPE" \
  --predictions_csv "$OUTPUT_CSV" \
  --output_json "$TEST_SUMMARY_JSON" \
  --input_path "$INPUT_PATH" \
  --train_data_path "$TRAIN_DATA_PATH" \
  --mem_guide "$MEM_GUIDE"

echo "Test complete: $OUTPUT_CSV"
echo "Score summary: $TEST_SUMMARY_JSON"
