#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "${1:-}" == "--help" ]]; then
  sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

# Usage:
#   bash scripts/evolve.sh
# Configure with environment variables; see README.md.

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
DATASET_TYPE="${DATASET_TYPE:-futurex}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/$DATASET_TYPE}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
PARALLELISM="${PARALLELISM:-1}"
MEMGUIDE_ROUNDS_PER_CYCLE="${MEMGUIDE_ROUNDS_PER_CYCLE:-2}"
EXPERIENCE_ROUNDS_PER_CYCLE="${EXPERIENCE_ROUNDS_PER_CYCLE:-1}"
EXPLORATION_OVER_EXPANSION="${EXPLORATION_OVER_EXPANSION:-1:1}"
MAX_TURNS="${MAX_TURNS:-2}"
SUBAGENT_MAX_TURNS="${SUBAGENT_MAX_TURNS:-10}"
SEARCH_PROVIDER="${SEARCH_PROVIDER:-tavily}"
SEARCH_MAX_RESULTS="${SEARCH_MAX_RESULTS:-1}"
SEARCH_MAX_CHARS_PER_RESULT="${SEARCH_MAX_CHARS_PER_RESULT:-30000}"
SEARCH_MAX_TOTAL_CHARS="${SEARCH_MAX_TOTAL_CHARS:-30000}"
SUMMARY_MAX_CHARS="${SUMMARY_MAX_CHARS:-90000}"
ENABLE_API_CACHE="${ENABLE_API_CACHE:-1}"
USE_RAW_CONTEXT="${USE_RAW_CONTEXT:-1}"

case "$DATASET_TYPE" in
  futurex)
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$REPO_DIR/data/FutureX/train-20of208.parquet}"
    VAL_DATA_PATH="${VAL_DATA_PATH:-$TRAIN_DATA_PATH}"
    ;;
  prophet_arena)
    TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$REPO_DIR/data/Prophet-arena/subset_data_Companies_train_5.csv}"
    VAL_DATA_PATH="${VAL_DATA_PATH:-$TRAIN_DATA_PATH}"
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
if [[ -z "${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}" ]]; then
  echo "Set OPENAI_API_KEY or OPENROUTER_API_KEY" >&2
  exit 2
fi
if [[ -z "${OPENAI_MODEL:-}" ]]; then
  echo "Set OPENAI_MODEL" >&2
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
if [[ ! -f "$TRAIN_DATA_PATH" || ! -f "$VAL_DATA_PATH" ]]; then
  echo "Training or validation data does not exist" >&2
  echo "TRAIN_DATA_PATH=$TRAIN_DATA_PATH" >&2
  echo "VAL_DATA_PATH=$VAL_DATA_PATH" >&2
  exit 2
fi

FACTUAL_MEMORY_DIR="$RUN_DIR/FactualMemory"
HISTORY_EVOLUTION_DIR="$RUN_DIR/HistoryEvolution"
HISTORY_ROLLOUT_DIR="$RUN_DIR/HistoryRollout"
MEMGUIDE_DIR="$RUN_DIR/MemGuide"
MEMTOOL_DIR="$RUN_DIR/MemTool"
EXPERIENCE_BANK_DIR="$RUN_DIR/ExperienceBank"
CACHE_DIR="$RUN_DIR/cache"

OPTIONAL_FLAGS=()
[[ "$ENABLE_API_CACHE" == "1" ]] && OPTIONAL_FLAGS+=(--enable_api_cache --disable_main_agent_final_answer_cache)
[[ "$USE_RAW_CONTEXT" == "1" ]] && OPTIONAL_FLAGS+=(--use_tavilty_raw_context)
[[ "$DATASET_TYPE" == "prophet_arena" ]] && OPTIONAL_FLAGS+=(--use_market_data_in_prophet_arena)

mkdir -p "$RUN_DIR"
cd "$REPO_DIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

echo "Starting ForeDreamer evolution"
echo "dataset=$DATASET_TYPE iterations=$NUM_ITERATIONS run_dir=$RUN_DIR model=$OPENAI_MODEL"
"$PYTHON_BIN" src/evolve_experience_memguide_and_memtool.py \
  --dataset_type "$DATASET_TYPE" \
  --train_data_path "$TRAIN_DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --num_iterations "$NUM_ITERATIONS" \
  --parallelism "$PARALLELISM" \
  --memguide_rounds_per_cycle "$MEMGUIDE_ROUNDS_PER_CYCLE" \
  --experience_rounds_per_cycle "$EXPERIENCE_ROUNDS_PER_CYCLE" \
  --experience_max_suggestions 2 \
  --max_turns "$MAX_TURNS" \
  --subagent_max_turns "$SUBAGENT_MAX_TURNS" \
  --search_provider "$SEARCH_PROVIDER" \
  --search_max_results "$SEARCH_MAX_RESULTS" \
  --search_max_chars_per_result "$SEARCH_MAX_CHARS_PER_RESULT" \
  --search_max_total_chars "$SEARCH_MAX_TOTAL_CHARS" \
  --summary_max_chars "$SUMMARY_MAX_CHARS" \
  --optimization_reuse_duplicated_tool \
  --optimization_encourage_exploration \
  --exploration_over_expansion "$EXPLORATION_OVER_EXPANSION" \
  --factual_memory_dir "$FACTUAL_MEMORY_DIR" \
  --history_evolution_dir "$HISTORY_EVOLUTION_DIR" \
  --history_rollout_dir "$HISTORY_ROLLOUT_DIR" \
  --memguide_dir "$MEMGUIDE_DIR" \
  --memtool_dir "$MEMTOOL_DIR" \
  --experience_bank_dir "$EXPERIENCE_BANK_DIR" \
  --cache_dir "$CACHE_DIR" \
  "${OPTIONAL_FLAGS[@]}"

"$PYTHON_BIN" src/select_best_assets.py \
  --dataset_type "$DATASET_TYPE" \
  --memguide_dir "$MEMGUIDE_DIR" \
  --experience_bank_dir "$EXPERIENCE_BANK_DIR" \
  > "$RUN_DIR/best_assets.json"

echo "Evolution complete: $RUN_DIR"
echo "Selected assets: $RUN_DIR/best_assets.json"
