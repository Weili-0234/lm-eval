#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname -- "${HARNESS_ROOT}")}"
LM_EVAL="${LM_EVAL:-${PROJECT_ROOT}/envs/lm-eval/bin/lm-eval}"
RESULTS_ROOT="${RESULTS_ROOT:-${HARNESS_ROOT}/results/endpoint}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache/huggingface}"

usage() {
  cat <<'EOF'
Usage: run_qat_endpoint_benchmark.sh MODEL TASK [LIMIT]

Models:
  qwen3-nvfp4 qwen35-nvfp4 qwen3-bf16 qwen35-bf16
  qwen3-q4k   qwen35-q4k

Tasks:
  ruler aime25_avg4 gpqa_diamond mmlu_pro humaneval ifeval
  pile_10k wikitext ppl

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
  *)
    echo "Unknown model: ${MODEL_KEY}" >&2
    usage >&2
    exit 2
    ;;
esac

case "${TASK_KEY}" in
  ruler|aime25_avg4|gpqa_diamond|mmlu_pro)
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
curl --max-time 5 -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

OUTPUT_SUFFIX="${TASK_KEY}"
LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  OUTPUT_SUFFIX="smoke/${TASK_KEY}-limit${LIMIT}"
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

COMMON_ARGS=(
  --batch_size 1
  --cache_requests refresh
  --seed 0
  --log_samples
  --output_path "${RESULTS_ROOT}/${RESULT_TAG}/${OUTPUT_SUFFIX}"
  "${LIMIT_ARGS[@]}"
)

completion_model_args() {
  printf '%s' \
    "model=${MODEL_ALIAS},base_url=http://127.0.0.1:${PORT}/v1/completions," \
    "tokenizer=${TOKENIZER},tokenizer_backend=huggingface," \
    "tokenized_requests=False,num_concurrent=${NUM_CONCURRENT}," \
    "timeout=${REQUEST_TIMEOUT},max_length=32768"
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
        printf '{"do_sample":true,"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"max_gen_toks":%s,"chat_template_kwargs":{"enable_thinking":true}}' "${max_gen_toks}"
        ;;
      nonthinking-coding)
        printf '{"do_sample":true,"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"max_gen_toks":%s,"add_generation_prompt":false,"continue_final_message":true,"chat_template_kwargs":{"enable_thinking":false}}' "${max_gen_toks}"
        ;;
      nonthinking)
        printf '{"do_sample":true,"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"max_gen_toks":%s,"chat_template_kwargs":{"enable_thinking":false}}' "${max_gen_toks}"
        ;;
    esac
  else
    case "${mode}" in
      thinking)
        printf '{"do_sample":true,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0,"max_gen_toks":%s,"chat_template_kwargs":{"enable_thinking":true}}' "${max_gen_toks}"
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

export HF_HOME TOKENIZERS_PARALLELISM=false
cd "${HARNESS_ROOT}"
mkdir -p "${RESULTS_ROOT}/${RESULT_TAG}/$(dirname -- "${OUTPUT_SUFFIX}")"
echo "${MODEL_KEY}/${TASK_KEY}: num_concurrent=${NUM_CONCURRENT}, timeout=${REQUEST_TIMEOUT}s"

case "${TASK_KEY}" in
  ruler)
    run_completion ruler --metadata '{"max_seq_lengths":[32768]}'
    ;;
  aime25_avg4)
    # -1 asks each server request to draw an independent random seed.
    run_chat aime25_avg4 thinking 30000 -1
    ;;
  gpqa_diamond)
    run_chat gpqa_diamond_cot_zeroshot thinking 30000
    ;;
  mmlu_pro)
    run_chat mmlu_pro thinking 8192
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
