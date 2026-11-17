# Kimi K3.1 32-card EP prefill recipe

Run launch_32gpu_ep.sh once on each of eight four-GPU nodes, passing rank 0..7.
Set SGLANG_ROOT, ADAPTER_ROOT, PLUGIN_ROOT, MODEL_PATH, GEMM_LIBPATH and
HEAD_ADDR (host:distributed-port); PORT defaults to 30116. Set the network
interfaces in the script for the target cluster before launch.

This archives the 1116 report: TP8/CP8/EP8/PP4, DeepGEMM W4A8, NCCL EP
transport, 8192-token prefill chunks, context132096, radix cache off,
max-running16, static0.90 and 144-GiB Torch budget. Physical cards are
BW1300_LC about144GiB; do not label this a96-GiB recipe.
The companion plugin kimi-k3.1-day0 branch supplies Kimi31 EP/checkpoint
hooks and SiTU contiguous quantization. Adapter needs no additional runtime
source changes beyond its existing cdb6e31 baseline.

Historical C1 result:131072 input/1 output,12.043s full HTTP response,
10883.38 input tokens/s. One warmup and one measured request; not strict
TTFT or a repeated statistical result. Native DeepEP Buffer failed; the
recipe explicitly selects the measured NCCL transport. No router candidate
or later decode/PP-loader optimization is included.

Verification for this archival delivery is syntax/configuration checking;
no additional GPU performance or quality run is claimed.
