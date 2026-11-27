import ast
import importlib.util
import sys
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('slimquant_native_test', Path(__file__).parents[2] / 'hcu_adapter/slimquant_native.py')
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


class NativeDenseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = {'compression_config': {'quant_method': 'slimquant_w4a8', 'mixed_precision': {'w8a8_include': {
            'weight_bits': 8, 'activation_bits': 8, 'format': 'int8-per-channel', 'modules': [
                'language_model.model.layers.*.self_attn.q_proj',
                'language_model.model.layers.*.self_attn.k_proj',
                'language_model.model.layers.*.self_attn.v_proj',
                'language_model.model.layers.*.self_attn.q_a_proj',
                'language_model.model.layers.*.self_attn.kv_a_proj_with_mqa',
                'language_model.model.layers.*.block_sparse_moe.routed_expert_up_proj',
                'language_model.model.layers.*.block_sparse_moe.shared_experts.gate_proj',
                'language_model.model.layers.*.block_sparse_moe.shared_experts.up_proj',
                'language_model.model.layers.0.mlp.gate_proj',
                'language_model.model.layers.0.mlp.up_proj']}}}}
        Path(self.directory.name, 'config.json').write_text(json.dumps(config))
        self.env = patch.dict(os.environ, {'HCU_KIMI31_NATIVE_DENSE_INT8': '1', 'HCU_KIMI31_DAY0': '1',
                                         'HCU_KIMI31_MODELSLIM_CHECKPOINT': self.directory.name})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_shim_preserves_raw_dense_payload_and_scale(self):
        source = Path(__file__).parents[2] / 'hcu_adapter/slimquant.py'
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.ClassDef) and node.name == '_SlimquantShard')
        namespace = {'__package__': 'hcu_adapter'}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        weight, scale = object(), object()
        key = 'model.layers.1.block_sparse_moe.routed_expert_up_proj.weight'
        class Shard:
            def get_tensor(self, name):
                return {key: weight, key + '_scale': scale}[name]
        proxy = namespace['_SlimquantShard'].__new__(namespace['_SlimquantShard'])
        proxy._active, proxy._f = True, Shard()
        with patch.dict(sys.modules, {'hcu_adapter.slimquant_native': N}):
            self.assertTrue(proxy._keep(key + '_scale'))
            self.assertIs(proxy.get_tensor(key), weight)
            self.assertIs(proxy.get_tensor(key + '_scale'), scale)

    def test_stacked_qkv_requires_complete_allowlist(self):
        self.assertTrue(N.native_dense_module('model.layers.2.self_attn.qkv_proj'))
        self.assertTrue(N.native_dense_module('model.layers.2.self_attn.fused_qkv_a_proj_with_mqa'))
        self.assertFalse(N.native_dense_module('model.layers.2.mlp.gate_up_proj'))

    def test_modelslim_moe_namespace_aliases_preserve_dense_layer_zero(self):
        pairs = (
            ('model.layers.2.mlp.shared_experts.gate_up_proj',
             'language_model.model.layers.2.block_sparse_moe.shared_experts.gate_up_proj'),
            ('model.layers.2.mlp.routed_expert_up_proj',
             'language_model.model.layers.2.block_sparse_moe.routed_expert_up_proj'),
        )
        for consumer, producer in pairs:
            self.assertTrue(N.native_dense_module(consumer), consumer)
            self.assertTrue(N.native_dense_module(producer), producer)
        self.assertTrue(N.native_dense_module('model.layers.0.mlp.gate_up_proj'))
        self.assertFalse(N.native_dense_module('model.layers.2.mlp.gate_up_proj'))
        self.assertFalse(N.native_dense_module('model.layers.2.mlp.gate'))

    def test_weight_and_scale_match_same_contract(self):
        prefix = 'language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj'
        self.assertTrue(N.native_dense_tensor(prefix + '.weight'))
        self.assertTrue(N.native_dense_tensor(prefix + '.weight_scale'))
        self.assertFalse(N.native_dense_tensor(prefix + '.weight_scale_inv'))
        self.assertFalse(N.native_dense_tensor('model.layers.1.block_sparse_moe.experts.0.w1.weight'))
        self.assertFalse(N.native_dense_tensor('model.layers.1.block_sparse_moe.gate.weight'))

    def test_full_rank_kda_qkv_readout_matches_mixed_merged_consumer(self):
        path = Path(self.directory.name, 'config.json')
        config = json.loads(path.read_text())
        config['text_config'] = {'linear_attn_config': {
            'use_full_rank_gate': True, 'kda_layers': [1, 2]}}
        path.write_text(json.dumps(config))
        N.native_dense_modules.cache_clear()
        N.native_dense_config.cache_clear()
        for leaf in ('q_proj', 'k_proj', 'v_proj', 'g_proj', 'fused_qkvg_proj'):
            prefix = 'model.layers.0.self_attn.' + leaf
            self.assertFalse(N.native_dense_module(prefix))
            self.assertFalse(N.native_dense_tensor(prefix + '.weight'))
            self.assertFalse(N.native_dense_tensor(prefix + '.weight_scale'))
        self.assertTrue(N.native_dense_module('model.layers.3.self_attn.q_a_proj'))
        self.assertTrue(N.native_dense_module('model.layers.0.block_sparse_moe.routed_expert_up_proj'))

    def test_baseline_flag_off_keeps_old_format(self):
        with patch.dict(os.environ, {'HCU_KIMI31_NATIVE_DENSE_INT8': '0'}):
            self.assertFalse(N.native_dense_module('model.layers.1.self_attn.qkv_proj'))

    def test_malformed_quant_metadata_rejected(self):
        Path(self.directory.name, 'config.json').write_text('{}')
        N.native_dense_modules.cache_clear()
        N.native_dense_config.cache_clear()
        with self.assertRaisesRegex(ValueError, 'metadata'):
            N.native_dense_module('model.layers.1.self_attn.qkv_proj')


if __name__ == '__main__':
    unittest.main()
