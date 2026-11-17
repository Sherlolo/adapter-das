#!/usr/bin/env bash
# TP8 / CP8 / EP8 / PP4

set -euo pipefail
# 每个容器各启动一次：bash launch_32gpu_ep.sh NODE_RANK
rank=${1:?请指定节点 rank}
[[ $rank =~ ^[0-7]$ ]] || { echo 'rank 必须为 0..7'; exit 2; }
pp=4
nnodes=$((pp * 2))
: "${SGLANG_ROOT:?set SGLANG_ROOT}" "${ADAPTER_ROOT:?set ADAPTER_ROOT}" "${PLUGIN_ROOT:?set PLUGIN_ROOT}"
: "${MODEL_PATH:?set MODEL_PATH}" "${GEMM_LIBPATH:?set GEMM_LIBPATH}" "${HEAD_ADDR:?set HEAD_ADDR}"
model=$MODEL_PATH
head=$HEAD_ADDR
port=${PORT:-30116}

export PYTHONPATH="$SGLANG_ROOT/python:$ADAPTER_ROOT:$PLUGIN_ROOT"
export PYTHONDONTWRITEBYTECODE=1 HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu HCU_KIMI31_DAY0=1
export HCU_MQA_RAGGED_REDUCE_HEADS=1 HCU_SLIMQUANT_FUSED_GROUP_QUANT=1
export HCU_KIMI31_DEEPEP=1 HCU_DEEPGEMM_BRIDGE=1
export HCU_PLATFORM_SEMANTICS=cuda HCU_KIMI31_EP_TRANSPORT=nccl
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1 SGLANG_USE_W4A8_MASKED_HIPC=1
export HCU_PREFILL_CP=1 KIMI_DEVICE_BUDGET_GIB=144
export NCCL_SOCKET_IFNAME=ibp129s0 GLOO_SOCKET_IFNAME=ibp129s0
if ((rank >= 4)); then export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0; fi
unset NCCL_NET NCCL_IB_DISABLE NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
if ((pp == 4)); then export NCCL_NET=Socket NCCL_IB_DISABLE=1; fi
export SGLANG_K3_ATTN_RESIDUAL_HCU=1 SGLANG_KDA_USE_HCU_OP=1
export SGLANG_K3_FUSED_FRONT=0 SGLANG_K3_SHARDED_MOE=0 SGLANG_K3_AR_FUSION=0
export SGLANG_OPT_USE_TOPK_V2=0 SGLANG_DSA_FUSE_TOPK=0 SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
unset HIPBLASLT_TUNING_OVERRIDE_FILE HIPBLASLT_TUNING_FILE ROCBLAS_UNSELECT_CHECK ROCBLAS_LAYER HIPBLASLT_LOG_MASK
export ROCBLAS_TENSILE_LIBPATH="$GEMM_LIBPATH"

cd "$PLUGIN_ROOT"
exec python3 -B -m hcu_sglang \
  --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm --quantization slimquant_w4a8_marlin \
  --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave --trust-remote-code --model-path "$model" --served-model-name kimi-k3.1 \
  --tp-size 8 --pp-size "$pp" --ep-size 8 --nnodes "$nnodes" --node-rank "$rank" \
  --dist-init-addr "$head" --dist-timeout 1800 --random-seed 42 --watchdog-timeout 1800 \
  --mem-fraction-static 0.90 --max-running-requests 16 --context-length 132096 --chunked-prefill-size "${CHUNK_SIZE:-8192}" \
  --weight-loader-prefetch-checkpoints --model-loader-extra-config '{"enable_multithread_load":false}' \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --mamba-full-memory-ratio 0.46 \
  --host 0.0.0.0 --port "$port" --skip-server-warmup --bf16-gemm-backend torch \
  --attention-backend dsa --dsa-topk-backend torch --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --linear-attn-backend triton --linear-attn-prefill-backend triton --page-size 64 --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 --mm-attention-backend fa3 \
  --disable-prefill-cuda-graph --disable-decode-cuda-graph --disable-overlap-schedule --disable-radix-cache
