# Kimi K3.1 DSpark 平台接口适配（2026-11-18）

用途：与独立 HCU SGLang plugin 组合，适配 SGLang 固定提交 `6321f714ddd9c5da511b0cfefa0ebabc4277cf36` 的平台辅助函数接口。adapter 基点为 `736a6837986aee5ae0373e4af76d4298839b9fea`。

改动范围：`platform_seam._replace_version_gates` 同时识别 lru_cache 包装的 partial 与直接引用同一 `_check_cuda_device_version` 的函数，并同步 utils 导出；避免 HCU CUDA 语义下对 `torch.version.cuda=None` 的访问。`sglang_spawn_device.install` 从 utils 公开导出读取 `maybe_reindex_device_id`，按函数实际 module 登记 hook，兼容 helper 移动。

实际验证：目标容器执行 feature-gate 与 spawn-device 定向回归，共 35 项通过；真实模型 preflight 的 9 个 CUDA feature gates 被置为 False。原始日志为本次工作目录的 `evidence/adapter-regression.log`、`evidence/preflight2-rank0.log`。本地解释器没有 pytest，测试在实际 HCU 容器执行。

组合边界：草稿 Speculators 配置、DSpark 注册、W4A8 masked SiTU、DeepEP 和 Graph layout 由 plugin 负责；本页不把 adapter 检查等同于完整模型精度或性能验收。用户仅要求少量请求检查，不运行数据集。完整服务结果以工作目录 REPORT.md 和实际请求原件为准。

此组合显式设置 `HCU_DEEPEP_BRIDGE=0`：plugin 自行匹配 HCU DeepEP 的 `quant_type` ABI，不能先由 adapter 将同一接口包装成 `use_fp8` 签名。重复包装曾导致完整模型 Graph warmup 得到不符合 W4A8 契约的激活；关闭后完整 adapter/plugin 环境下16-rank INT8 dispatch Graph 的三次动态输入/路由重放均最大误差0。DeepGEMM bridge 保持启用；此配置不改变 TP collective。

2026-11-19 DP4 无 decode graph 验证时，空闲 DP rank 的 eager target-verify 会向 DSA metadata planner 传入 `(0, 1)` context lengths。LightOp `get_paged_mqa_logits_metadata` 对 aligned batch 0 抛错，导致 rank3 首个请求退出。adapter 现仅对零行返回同设备 `(1, 2)` int32 零占位；非空仍调用 LightOp planner。上游 DSA indexer 对 idle/空序列已提前返回，不消费该占位。目标容器以 Python 3.12、`-S` 和指向本仓的 `PYTHONPATH` 执行 `tests/kernels/test_paged_mqa_metadata.py`，5 项通过；完整服务恢复与性能仍以重新启动后的实测为准。

2026-11-19 合并验证：spawn helper优先使用公开导出，兼容device/common旧路径，登记实际module；保留partial与function形式feature gates。远程隐藏GPU的CPU定向回归41 passed、2项GPU用例skipped，包含新增三路径安装测试。本轮未启动GPU服务或探针。
