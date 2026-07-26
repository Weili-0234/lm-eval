#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname -- "${HARNESS_ROOT}")}"
LM_EVAL="${LM_EVAL:-${PROJECT_ROOT}/envs/lm-eval/bin/lm-eval}"
RESULTS_ROOT="${RESULTS_ROOT:-${HARNESS_ROOT}/results/endpoint}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"

# Provenance guards: a formal run must come from a clean checkout of the pinned
# harness commit. Uncommitted edits under lm_eval/ or scripts/ silently change
# task definitions, subsets, or invocation args while every check still passes.
HARNESS_SHA="$(git -C "${HARNESS_ROOT}" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
export HARNESS_SHA
if [[ -n "$(git -C "${HARNESS_ROOT}" status --porcelain -- lm_eval scripts 2>/dev/null)" \
      && "${HARNESS_DIRTY_OK:-0}" != "1" ]]; then
  echo "Harness working tree is dirty under lm_eval/ or scripts/ (${HARNESS_ROOT})." >&2
  echo "Commit + push the change and bump the skill's SHA pin instead of running from" >&2
  echo "a local patch. Set HARNESS_DIRTY_OK=1 only for a deliberate run that will be" >&2
  echo "filed with a 'deviation:' note." >&2
  exit 2
fi
if [[ -n "${EXPECTED_HARNESS_SHA:-}" \
      && "${EXPECTED_HARNESS_SHA}" != "${HARNESS_SHA}"* \
      && "${HARNESS_SHA}" != "${EXPECTED_HARNESS_SHA}"* ]]; then
  echo "Harness checkout ${HARNESS_SHA} does not match EXPECTED_HARNESS_SHA=${EXPECTED_HARNESS_SHA}." >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Usage: run_qat_endpoint_benchmark.sh MODEL TASK [LIMIT]

Models:
  qwen3-nvfp4 qwen35-nvfp4 qwen3-bf16 qwen35-bf16
  qwen3-q4k   qwen35-q4k
  custom      (generic vLLM endpoint; requires env FAMILY={qwen3|qwen35},
               MODEL_ALIAS, TOKENIZER, PORT_OVERRIDE, RESULT_TAG_OVERRIDE)

Tasks:
  ruler ruler128k aime25_avg4 gpqa_diamond mmlu_pro humaneval ifeval
  pile_10k wikitext ppl

ruler128k runs the three quantization-discriminative RULER tasks
(ruler_qa_squad, niah_multikey_3, ruler_qa_hotpot) at a 131,072-token
sequence length against a dedicated 128K server profile (131,328 capacity).

LIMIT is an optional positive integer for smoke tests.
EOF
}

[[ $# -ge 2 && $# -le 3 ]] || {
  usage >&2
  exit 2
}

MODEL_KEY="$1"
TASK_KEY="$2"
LIMIT="${3:-}"

case "${MODEL_KEY}" in
  qwen3-nvfp4)
    FAMILY=qwen3
    BACKEND=vllm
    MODEL_ALIAS=qwen3-8b-nvfp4-step6242
    PORT=8000
    RESULT_TAG=qwen3-8b-nvfp4
    TOKENIZER="${PROJECT_ROOT}/model-and-data/Qwen3-8B-BF16"
    ;;
  qwen35-nvfp4)
    FAMILY=qwen35
    BACKEND=vllm
    MODEL_ALIAS=qwen3.5-9b-nvfp4-step5264
    PORT=8001
    RESULT_TAG=qwen3.5-9b-nvfp4
    TOKENIZER="${PROJECT_ROOT}/model-and-data/Qwen3.5-9B-BF16"
    ;;
  qwen3-bf16)
    FAMILY=qwen3
    BACKEND=vllm
    MODEL_ALIAS=qwen3-8b-bf16
    PORT=8002
    RESULT_TAG=qwen3-8b-bf16
    TOKENIZER="${PROJECT_ROOT}/model-and-data/Qwen3-8B-BF16"
    ;;
  qwen35-bf16)
    FAMILY=qwen35
    BACKEND=vllm
    MODEL_ALIAS=qwen3.5-9b-bf16
    PORT=8003
    RESULT_TAG=qwen3.5-9b-bf16
    TOKENIZER="${PROJECT_ROOT}/model-and-data/Qwen3.5-9B-BF16"
    ;;
  qwen3-q4k)
    FAMILY=qwen3
    BACKEND=llamacpp
    MODEL_ALIAS=qwen3-8b-q4k-step6242
    PORT=8081
    RESULT_TAG=qwen3-8b-q4k
    TOKENIZER="${PROJECT_ROOT}/model-and-data/Qwen3-8B-BF16"
    ;;
  qwen35-q4k)
    FAMILY=qwen35
    BACKEND=llamacpp
    MODEL_ALIAS=qwen3.5-9b-q4k-step5264
    PORT=8082
    RESULT_TAG=qwen3.5-9b-q4k
    TOKENIZER="${PROJECT_ROOT}/model-and-data/Qwen3.5-9B-BF16"
    ;;
  custom)
    # Generic vLLM endpoint for the standardized eval pipeline: identity comes
    # entirely from the environment so any checkpoint/alias/port works.
    FAMILY="${FAMILY:?custom model requires FAMILY=qwen3|qwen35}"
    BACKEND=vllm
    MODEL_ALIAS="${MODEL_ALIAS:?custom model requires MODEL_ALIAS}"
    PORT="${PORT_OVERRIDE:?custom model requires PORT_OVERRIDE}"
    RESULT_TAG="${RESULT_TAG_OVERRIDE:?custom model requires RESULT_TAG_OVERRIDE}"
    TOKENIZER="${TOKENIZER:?custom model requires TOKENIZER (family BF16 tokenizer dir)}"
    [[ "${FAMILY}" == qwen3 || "${FAMILY}" == qwen35 ]] || {
      echo "FAMILY must be qwen3 or qwen35, got: ${FAMILY}" >&2
      exit 2
    }
    # FAMILY selects the entire sampling recipe, the MMLU-Pro generation
    # budget, and 128K rope handling. When the caller can point at the served
    # checkpoint (MODEL_PATH), verify the assertion against its config.json
    # instead of trusting a hand-typed value (qwen3_5 -> qwen35; qwen3 -> qwen3).
    if [[ -n "${MODEL_PATH:-}" && -f "${MODEL_PATH}/config.json" ]]; then
      CFG_FAMILY="$(python3 -c '
import json, sys
mt = json.load(open(sys.argv[1])).get("model_type", "")
if "qwen3_5" in mt or "qwen3.5" in mt:
    print("qwen35")
elif mt.startswith("qwen3"):
    print("qwen3")
else:
    print("unknown")' "${MODEL_PATH}/config.json")"
      if [[ "${CFG_FAMILY}" != "unknown" && "${CFG_FAMILY}" != "${FAMILY}" ]]; then
        echo "FAMILY=${FAMILY} contradicts ${MODEL_PATH}/config.json (model_type => ${CFG_FAMILY})." >&2
        exit 2
      fi
    fi
    ;;
  *)
    echo "Unknown model: ${MODEL_KEY}" >&2
    usage >&2
    exit 2
    ;;
esac

PORT="${PORT_OVERRIDE:-${PORT}}"
RESULT_TAG="${RESULT_TAG_OVERRIDE:-${RESULT_TAG}}"
# Standardized protocol: the frozen, revision-pinned 1k chat subset. The
# legacy full-12k 5-shot group must be an explicit, labeled deviation — never
# a forgotten env var (it exits 0 and files under the same output name).
MMLU_TASKS="${MMLU_TASKS:-mmlu_pro_chat}"
if [[ "${MMLU_TASKS}" != "mmlu_pro_chat" && "${MMLU_TASKS_LEGACY_OK:-0}" != "1" ]]; then
  echo "MMLU_TASKS=${MMLU_TASKS} is not the standardized mmlu_pro_chat subset." >&2
  echo "Set MMLU_TASKS_LEGACY_OK=1 only for a deliberate legacy run filed with a" >&2
  echo "'deviation:' note." >&2
  exit 2
fi

case "${TASK_KEY}" in
  ruler|ruler128k|aime25_avg4|gpqa_diamond|mmlu_pro)
    DEFAULT_CONCURRENT=8
    ;;
  humaneval)
    DEFAULT_CONCURRENT=16
    ;;
  ifeval|pile_10k|wikitext|ppl)
    DEFAULT_CONCURRENT=32
    ;;
  *)
    echo "Unknown task: ${TASK_KEY}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${BACKEND}" == llamacpp && "${DEFAULT_CONCURRENT}" -gt 8 ]]; then
  DEFAULT_CONCURRENT=8
fi

NUM_CONCURRENT="${NUM_CONCURRENT:-${DEFAULT_CONCURRENT}}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-7200}"
[[ "${NUM_CONCURRENT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_CONCURRENT must be a positive integer." >&2
  exit 2
}
[[ "${REQUEST_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "REQUEST_TIMEOUT must be a positive integer." >&2
  exit 2
}
[[ "${PORT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PORT_OVERRIDE must be a valid positive port number." >&2
  exit 2
}
if [[ -n "${LIMIT}" && ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer." >&2
  exit 2
fi

[[ -x "${LM_EVAL}" ]] || {
  echo "lm-eval is not installed at ${LM_EVAL}." >&2
  exit 1
}
[[ -d "${TOKENIZER}" ]] || {
  echo "Tokenizer directory not found: ${TOKENIZER}" >&2
  exit 1
}
# The tokenizer identity shapes PPL rolling-loglikelihood windows and the
# RULER 131,072-token haystacks. When the caller pins a content hash, assert
# it — a directory path alone does not identify a tokenizer.
if [[ -n "${TOKENIZER_SHA256:-}" ]]; then
  ACTUAL_TOK_SHA="$(sha256sum "${TOKENIZER}/tokenizer.json" | cut -d' ' -f1)"
  [[ "${ACTUAL_TOK_SHA}" == "${TOKENIZER_SHA256}" ]] || {
    echo "tokenizer.json sha256 mismatch at ${TOKENIZER}:" >&2
    echo "  expected ${TOKENIZER_SHA256}" >&2
    echo "  actual   ${ACTUAL_TOK_SHA}" >&2
    exit 2
  }
fi
curl --max-time 5 -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

OUTPUT_SUFFIX="${OUTPUT_SUFFIX_OVERRIDE:-${TASK_KEY}}"
LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  OUTPUT_SUFFIX="smoke/${OUTPUT_SUFFIX}-limit${LIMIT}"
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

COMMON_ARGS=(
  --batch_size 1
  --cache_requests refresh
  # Endpoint evaluations are CPU-only; skip torch RNG initialization.
  --seed 0,1234,None,1234
  --log_samples
  --output_path "${RESULTS_ROOT}/${RESULT_TAG}/${OUTPUT_SUFFIX}"
  "${LIMIT_ARGS[@]}"
)

# 131072 for ruler128k; 32768 for everything else.
COMPLETION_MAX_LENGTH=32768

completion_model_args() {
  printf '%s' \
    "model=${MODEL_ALIAS},base_url=http://127.0.0.1:${PORT}/v1/completions," \
    "tokenizer=${TOKENIZER},tokenizer_backend=huggingface," \
    "tokenized_requests=False,num_concurrent=${NUM_CONCURRENT}," \
    "timeout=${REQUEST_TIMEOUT},max_length=${COMPLETION_MAX_LENGTH}"
}

chat_model_args() {
  local api_seed="$1"
  printf '%s' \
    "model=${MODEL_ALIAS},base_url=http://127.0.0.1:${PORT}/v1/chat/completions," \
    "tokenized_requests=False,num_concurrent=${NUM_CONCURRENT}," \
    "timeout=${REQUEST_TIMEOUT},max_length=32768,seed=${api_seed}"
}

generation_json() {
  local mode="$1"
  local max_gen_toks="$2"

  if [[ "${FAMILY}" == qwen3 ]]; then
    case "${mode}" in
      thinking)
        printf '{"do_sample":true,"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"max_gen_toks":%s,"chat_template_kwargs":{"enable_thinking":true}}' "${max_gen_toks}"
        ;;
      thinking-mmlu)
        printf '{"do_sample":true,"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"max_gen_toks":%s,"until":["<|im_end|>"],"chat_template_kwargs":{"enable_thinking":true}}' "${max_gen_toks}"
        ;;
      nonthinking-coding)
        printf '{"do_sample":true,"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"max_gen_toks":%s,"add_generation_prompt":false,"continue_final_message":true,"chat_template_kwargs":{"enable_thinking":false}}' "${max_gen_toks}"
        ;;
      nonthinking)
        printf '{"do_sample":true,"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"max_gen_toks":%s,"chat_template_kwargs":{"enable_thinking":false}}' "${max_gen_toks}"
        ;;
    esac
  else
    case "${mode}" in
      thinking)
        printf '{"do_sample":true,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0,"max_gen_toks":%s,"chat_template_kwargs":{"enable_thinking":true}}' "${max_gen_toks}"
        ;;
      thinking-mmlu)
        printf '{"do_sample":true,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0,"max_gen_toks":%s,"until":["<|im_end|>"],"chat_template_kwargs":{"enable_thinking":true}}' "${max_gen_toks}"
        ;;
      nonthinking-coding)
        printf '{"do_sample":true,"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"max_gen_toks":%s,"add_generation_prompt":false,"continue_final_message":true,"chat_template_kwargs":{"enable_thinking":false}}' "${max_gen_toks}"
        ;;
      nonthinking)
        printf '{"do_sample":true,"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0,"max_gen_toks":%s,"chat_template_kwargs":{"enable_thinking":false}}' "${max_gen_toks}"
        ;;
    esac
  fi
}

run_chat() {
  local task="$1"
  local mode="$2"
  local max_gen_toks="$3"
  local api_seed="${4:-1234}"
  local gen_kwargs
  gen_kwargs="$(generation_json "${mode}" "${max_gen_toks}")"

  "${LM_EVAL}" run \
    --model local-chat-completions \
    --model_args "$(chat_model_args "${api_seed}")" \
    --tasks "${task}" \
    --apply_chat_template \
    --gen_kwargs "${gen_kwargs}" \
    "${COMMON_ARGS[@]}"
}

run_completion() {
  local task="$1"
  shift
  "${LM_EVAL}" run \
    --model local-completions \
    --model_args "$(completion_model_args)" \
    --tasks "${task}" \
    "$@" \
    "${COMMON_ARGS[@]}"
}

export HF_HOME TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="" USE_TORCH=0
cd "${HARNESS_ROOT}"
mkdir -p "${RESULTS_ROOT}/${RESULT_TAG}/$(dirname -- "${OUTPUT_SUFFIX}")"
echo "${MODEL_KEY}/${TASK_KEY}: num_concurrent=${NUM_CONCURRENT}, timeout=${REQUEST_TIMEOUT}s"

case "${TASK_KEY}" in
  ruler)
    run_completion ruler --metadata '{"max_seq_lengths":[32768]}'
    ;;
  ruler128k)
    # Requires the dedicated 128K server profile (--max-model-len 131328;
    # Qwen3 additionally needs official static YaRN 4x, Qwen3.5 does not).
    COMPLETION_MAX_LENGTH=131072
    run_completion ruler_qa_squad,niah_multikey_3,ruler_qa_hotpot \
      --metadata '{"max_seq_lengths":[131072]}'
    ;;
  aime25_avg4)
    # -1 asks each server request to draw an independent random seed.
    run_chat aime25_avg4 thinking 30000 -1
    ;;
  gpqa_diamond)
    run_chat gpqa_diamond_cot_zeroshot thinking 30000
    ;;
  mmlu_pro)
    if [[ "${FAMILY}" == qwen35 ]]; then
      # Qwen3.5 frequently needs more than 8K thinking tokens before its final
      # choice. The largest measured five-shot prompt is 2,738 tokens, so 30K
      # generation remains within the 33,024-token serving limit. Override the
      # task's "Question:" stop string because Qwen3.5 naturally emits phrases
      # such as "Analyze the Question:" during reasoning.
      run_chat "${MMLU_TASKS}" thinking-mmlu 30000
    else
      run_chat "${MMLU_TASKS}" thinking-mmlu 8192
    fi
    ;;
  humaneval)
    export HF_ALLOW_CODE_EVAL=1
    COMMON_ARGS+=(--confirm_run_unsafe_code)
    run_chat humaneval_instruct nonthinking-coding 8192
    ;;
  ifeval)
    run_chat ifeval nonthinking 1280
    ;;
  pile_10k|wikitext)
    [[ "${BACKEND}" == vllm ]] || {
      echo "${MODEL_KEY}: ${TASK_KEY} is N/A because llama.cpp lacks echo logprobs." >&2
      exit 3
    }
    run_completion "${TASK_KEY}"
    ;;
  ppl)
    [[ "${BACKEND}" == vllm ]] || {
      echo "${MODEL_KEY}: PPL is N/A because llama.cpp lacks echo logprobs." >&2
      exit 3
    }
    run_completion pile_10k,wikitext
    ;;
esac
