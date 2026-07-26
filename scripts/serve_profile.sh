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

Env: VLLM (vllm binary, default `vllm` on PATH), SERVE_LOG (log file path),
     SERVED_NAME (--served-model-name; default basename(MODEL_PATH) — must
     equal the runner's MODEL_ALIAS or every request 404s).
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
  # One legitimate mismatch: an A4-schema export served as w4a16 on SM90
  # (H100 has no FP4-activation kernels, so vLLM realizes weight-only there —
  # the label matches what actually runs). Everything else is a mislabel.
  COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
  if [[ "${DTYPE}" == "w4a16" && "${SCHEMA_DTYPE}" == "w4a4" && "${COMPUTE_CAP}" == 9.* ]]; then
    echo "note: w4a16 realized via SM90 hardware fallback on an A4-schema export" >&2
  else
    echo "DTYPE=${DTYPE} contradicts the checkpoint schema (=> ${SCHEMA_DTYPE}):" >&2
    echo "  w4a4 needs an activation-quantized (A4) schema export; w4a16 needs the" >&2
    echo "  weight-only schema export (or SM90 hardware fallback); bf16 needs no" >&2
    echo "  quantization_config." >&2
    exit 2
  fi
fi

SERVED_NAME="${SERVED_NAME:-$(basename "${MODEL_PATH}")}"
CMD=("${VLLM_BIN}" serve "${MODEL_PATH}"
     --host 127.0.0.1 --port "${PORT}"
     --served-model-name "${SERVED_NAME}"
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

# Mirror the two engine-env mitigations Slot.start_server applies — without
# them the mandated manual path reintroduces both documented silent
# engine-death modes:
# 1. FlashInfer's JIT spawns bare `ninja`; if vllm lives in a venv, its bin
#    dir must be on PATH or the engine core dies with "Failed core proc(s)".
if [[ "${VLLM_BIN}" == */* ]]; then
  export PATH="$(cd "$(dirname "${VLLM_BIN}")" && pwd)${PATH:+:${PATH}}"
fi
# 2. Concurrent engines sharing one compile/autotune cache (or MXFP4/NVFP4
#    models sharing tuned-kernel entries) silently crash the engine core —
#    isolate the cache per (model, port).
RUN_DIR="$(dirname "${SERVE_LOG:-/tmp/serve_profile.log}")"
export VLLM_CACHE_ROOT="${RUN_DIR}/vllm-cache-$(basename "${MODEL_PATH}")-p${PORT}"
mkdir -p "${VLLM_CACHE_ROOT}"

# Keep the sidecar out of the harness working tree (a stray file under
# scripts/ would trip the runner's dirty-checkout guard).
SIDECAR="${RUN_DIR}/serving-manual-p${PORT}.json"
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
