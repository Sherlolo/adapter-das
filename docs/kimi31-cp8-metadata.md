# Kimi31 CP8 分页 MQA metadata 设备适配

用途：记录 SGLang `6321f714`、adapter `bcc63ef1`、gfx938/64CU 组合在2026-11-20发现的 decode metadata 接口问题。开发入口为 [hcu-adapter-dev](../skills/hcu-adapter-dev/SKILL.md)；规则见 [治理](governance.md)。

## 问题与修复

CP8/EP8/PP2 服务首个64-token请求进入 decode metadata 后，在 `IndexerMetadataKernel::run` 的 TensorMatcher 报 `Device value [rocm:0] not in the allowed options: [cuda]`。源文件 `paged_mqa_metadata.cuh` 的 `device.set_options<kDLCUDA>()` 与真实 DLPack 设备类型不符。此错误不同于先前 DeepEP 的 invalid device pointer；不能据此宣称通信修复失败。

`jit_dpskmeta` 仅在 `jit_sources` 的私有内容寻址编译输入中把该声明改成上游已有 `kDLGPU`，由公共头在 HIP 平台解析成 `kDLROCM`。没有修改 SGLang 文件或改写 Python 模块来源。新增 needle 同步 `TEXT_PATCH_ALLOWLIST`，变换实现参与缓存 key；不删除或覆盖旧 JIT 缓存。

## 检查与边界

`tests/adapter/test_jit_sources.py::test_paged_mqa_metadata_accepts_runtime_gpu_device` 在 offline 与固定社区 Git 资产两种夹具中，验证私有产物设备约束与原始资产不变。修复前两种夹具均失败；本轮完整 JIT 资产测试的最终结果以私有任务日志为准。

实卡通过原 `get_paged_mqa_logits_metadata` 入口，使用实际64个CU，覆盖 batch 1/2/8/65/2048/2049/4097；eager 和每种两次变输入 Graph replay 均与独立 CPU 整数参考完全一致。覆盖 tiny、small、多block分支及跨256-token边界。原始 `.cuh` SHA256 在执行前后保持一致。

上述证据只证明 metadata 数值与 Graph 设备接口；完整模型恢复、输出质量、128k并发性能和16rank Prof仍需单独验证。原始失败请求、四节点日志、CPU反例与实卡状态保存在本次私有任务目录，未把历史模型结果当成本次验收。
