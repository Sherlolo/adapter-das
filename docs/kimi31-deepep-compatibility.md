# Kimi K3.1 DeepEP compatibility

Preserve the vendor quant_type dispatch API and its introspectable signature
while retaining upstream use_fp8/BF16 translation and caller-frame routing
weights. Native positional/keyword calls pass through without dtype changes.
Repeated installation does not wrap the dual-API bridge twice.

Current SGLang startup also requires the updated CUDA-version gate recognition
and the moved maybe_reindex_device_id import, with legacy-location fallback.
No official SGLang source or vendor DeepEP kernels are modified.

Validated with the paired plugin native DeepEP low-latency/full Graph change
on unchanged official SGLang 8f2e5fe54efabf30a87138c6e6477676412686d4.
Complete Kimi checkpoint TP16/EP16/attention-DP2 loaded, full decode Graph
captured on all 16 ranks, and four deterministic short chat checks passed.
The plugin commit records the exact configuration and acceptance boundary.

CPU checks:

    python3 -B -S tests/adapter/test_kimi31_deepep_dual_api.py
    python3 -B -S tests/adapter/test_kimi31_startup_compat.py

The checks cover native INT8, old FP8/BF16, caller-frame routing weights,
signature visibility, old/new CUDA gate wrappers and GPU-selector locations.
TP-only AITER whole-model execution, 32-GPU scaling, performance and dataset
accuracy were not evaluated in this compatibility change.

Release checks: all six dispatch API tests and both startup compatibility tests passed on 2026-11-18.
