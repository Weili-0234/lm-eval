#!/usr/bin/env bash
# Pinned vLLM server launcher for the standardized QATFactory eval protocol.
#
# This is the ONLY sanctioned way to start an eval endpoint outside
# eval_queue.py. It emits byte-for-byte the same serve command the queue's
# Slot.start_server uses, validates that the checkpoint's quantization schema
# matches the declared inference dtype, cross-checks the model family against
# config.json, and writes a serving-manual.json sidecar for provenance.
# Hand-written `vllm serve` commands are forbidden by the evaluation skill:
# a missing --generation-config vllm or a wrong rope-scaling blob passes
# /health and silently moves scores.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: serve_profile.sh MODEL_PATH FAMILY DTYPE PROFILE PORT [GPUS]

  MODEL_PATH  checkpoint directory (must contain config.json)
  FAMILY      qwen3 | qwen35   (verified against config.json model_type)
  DTYPE       bf16 | w4a4 | w4a16   (verified against quantization_config:
              w4a4 requires input_activations in the schema; w4a16 requires a
              weight-only schema; bf16 requires no quantization_config)
  PROFILE     32k | ppl-safe | 128k
  PORT        TCP port to serve on
  GPUS        CUDA_VISIBLE_DEVICES value (default: 0); N comma-separated GPUs
              enable --tensor-parallel-size N

Env: VLLM (vllm binary, default `vllm` on PATH), SERVE_LOG (log file path).
EOF
}

[[ $# -ge 5 ]] || { usage >&2; exit 2; }
MODEL_PATH="$1"; FAMILY="$2"; DTYPE="$3"; PROFILE="$4"; PORT="$5"; GPUS="${6:-0}"
VLLM_BIN="${VLLM:-vllm}"

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "no config.json under ${MODEL_PATH}" >&2; exit 2; }
[[ "${FAMILY}" == qwen3 || "${FAMILY}" == qwen35 ]] || { echo "FAMILY must be qwen3|qwen35" >&2; exit 2; }
[[ "${DTYPE}" == bf16 || "${DTYPE}" == w4a4 || "${DTYPE}" == w4a16 ]] || { echo "DTYPE must be bf16|w4a4|w4a16" >&2; exit 2; }

case "${PROFILE}" in
  32k)      MAX_LEN=33024;  BATCHED=8192 ;;
  ppl-safe) MAX_LEN=33024;  BATCHED=4096 ;;
  128k)     MAX_LEN=131328; BATCHED=8192 ;;
  *) echo "PROFILE must be 32k|ppl-safe|128k" >&2; exit 2 ;;
esac

# family <-> config.json model_type
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
  echo "FAMILY=${FAMILY} contradicts config.json model_type (=> ${CFG_FAMILY})" >&2
  exit 2
fi

# dtype <-> quantization schema. The serving kernel path follows the
# checkpoint schema (input_activations quantized => W4A4 on SM100), so the
# declared dtype must match the schema or the row is mislabeled.
SCHEMA_DTYPE="$(python3 -c '
import json, sys
qc = json.load(open(sys.argv[1])).get("quantization_config")
if not qc:
    print("bf16")
else:
    has_act = any((g or {}).get("input_activations") for g in qc.get("config_groups", {}).values())
    print("w4a4" if has_act else "w4a16")' "${MODEL_PATH}/config.json")"
if [[ "${SCHEMA_DTYPE}" != "${DTYPE}" ]]; then
  echo "DTYPE=${DTYPE} contradicts the checkpoint schema (=> ${SCHEMA_DTYPE}):" >&2
  echo "  w4a4 needs an activation-quantized (A4) schema export; w4a16 needs the" >&2
  echo "  weight-only schema export; bf16 needs no quantization_config." >&2
  exit 2
fi

CMD=("${VLLM_BIN}" serve "${MODEL_PATH}"
     --host 127.0.0.1 --port "${PORT}"
     --served-model-name "$(basename "${MODEL_PATH}")"
     --max-model-len "${MAX_LEN}"
     --gpu-memory-utilization 0.85
     --generation-config vllm
     --max-num-batched-tokens "${BATCHED}")
NGPU="$(awk -F, '{print NF}' <<<"${GPUS}")"
[[ "${NGPU}" -gt 1 ]] && CMD+=(--tensor-parallel-size "${NGPU}")
[[ "${DTYPE}" == bf16 ]] && CMD+=(--dtype bfloat16)
if [[ "${PROFILE}" == 128k && "${FAMILY}" == qwen3 ]]; then
  # Qwen3-8B native context is 32K; official static YaRN 4x. Qwen3.5 is
  # 262K-native and must NOT get rope-scaling.
  CMD+=(--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}')
fi

SIDECAR="$(dirname "${SERVE_LOG:-./serve.log}")/serving-manual-p${PORT}.json"
python3 -c '
import json, subprocess, sys
model_path, family, dtype, profile, port, gpus, vllm_bin, sidecar = sys.argv[1:9]
try:
    version = subprocess.check_output([vllm_bin, "--version"], text=True, timeout=120).strip().splitlines()[-1]
except Exception:
    version = "unknown"
json.dump({"engine": "vllm", "engine_version": version, "engine_binary": vllm_bin,
           "model_path": model_path, "family": family, "inference_dtype": dtype,
           "profile": profile, "port": int(port), "gpus": gpus,
           "launcher": "serve_profile.sh"}, open(sidecar, "w"), indent=2)
' "${MODEL_PATH}" "${FAMILY}" "${DTYPE}" "${PROFILE}" "${PORT}" "${GPUS}" "${VLLM_BIN}" "${SIDECAR}"
echo "serving-manual sidecar: ${SIDECAR}"
echo "exec: CUDA_VISIBLE_DEVICES=${GPUS} ${CMD[*]}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
if [[ -n "${SERVE_LOG:-}" ]]; then
  exec "${CMD[@]}" >"${SERVE_LOG}" 2>&1
else
  exec "${CMD[@]}"
fi
