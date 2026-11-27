import fnmatch
import json
import os
from functools import lru_cache
from pathlib import Path


def native_dense_enabled():
    return (os.environ.get("HCU_KIMI31_NATIVE_DENSE_INT8") == "1"
            and os.environ.get("HCU_KIMI31_DAY0") == "1")


@lru_cache(maxsize=4)
def native_dense_config(checkpoint):
    return json.loads((Path(checkpoint) / "config.json").read_text())


@lru_cache(maxsize=4)
def native_dense_modules(checkpoint):
    config = native_dense_config(checkpoint)
    compression = config.get("compression_config", {})
    if compression.get("quant_method") != "slimquant_w4a8":
        raise ValueError("Native dense INT8 requires ModelSlim slimquant_w4a8 metadata")
    include = compression.get("mixed_precision", {}).get("w8a8_include", {})
    if (include.get("weight_bits") != 8 or include.get("activation_bits") != 8
            or include.get("format") != "int8-per-channel"):
        raise ValueError("Native dense INT8 requires explicit int8-per-channel w8a8_include")
    return tuple(include.get("modules", ()))


def native_dense_module(prefix):
    if not native_dense_enabled():
        return False
    checkpoint = os.environ.get("HCU_KIMI31_MODELSLIM_CHECKPOINT")
    if not checkpoint:
        raise ValueError("Native dense INT8 requires HCU_KIMI31_MODELSLIM_CHECKPOINT")
    if ".layers." not in prefix:
        return False
    prefix = "language_model.model.layers." + prefix.split(".layers.", 1)[1]
    for namespace in ("shared_experts", "routed_expert_up_proj", "routed_expert_down_proj"):
        prefix = prefix.replace(".mlp." + namespace, ".block_sparse_moe." + namespace)
    parent, leaf = prefix.rsplit(".", 1)
    config = native_dense_config(checkpoint)
    text = config.get("text_config", config)
    linear = text.get("linear_attn_config") or {}
    layer_idx = int(prefix.split(".layers.", 1)[1].split(".", 1)[0])
    if (leaf in ("q_proj", "k_proj", "v_proj", "g_proj", "qkv_proj")
            and parent.endswith(".self_attn")
            and layer_idx + 1 in linear.get("kda_layers", ())
            and linear.get("use_full_rank_gate", False)):
        leaf = "fused_qkvg_proj"
    components = {
        "fused_qkvg_proj": ("q_proj", "k_proj", "v_proj", "g_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "fused_qkv_a_proj_with_mqa": ("q_a_proj", "kv_a_proj_with_mqa"),
    }.get(leaf, (leaf,))
    patterns = native_dense_modules(checkpoint)
    return all(any(fnmatch.fnmatchcase(parent + "." + part, pattern)
                   for pattern in patterns) for part in components)


def native_dense_tensor(name):
    suffix = ".weight_scale" if name.endswith(".weight_scale") else ".weight"
    return name.endswith(suffix) and native_dense_module(name[:-len(suffix)])
