import ast
import functools
import inspect
import logging
from pathlib import Path
import sys
import types
import unittest

source=Path(__file__).resolve().parents[2]/'hcu_adapter/bridges/sglang.py'
tree=ast.parse(source.read_text())

class DualApi(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def original(self_,x,topk_idx,topk_weight,num_max_dispatch_tokens_per_rank,num_experts,quant_type=1,**kwargs):
            value=(x,topk_idx,topk_weight,num_max_dispatch_tokens_per_rank,num_experts,quant_type,kwargs)
            self.calls.append(value)
            return value
        self.Tensor=type('Tensor',(),{})
        ns={'_ll_orig':original,'_sys':sys,'_functools':functools,'torch':types.SimpleNamespace(Tensor=self.Tensor),'logging':logging}
        for name in ['_ll_dispatch_v0521','_ll_dispatch_dual']:
            node=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name)
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        self.dispatch=ns['_ll_dispatch_dual']
    def test_native_positional_int8(self):
        result=self.dispatch(None,'x','ids','weights',8,896,quant_type=1,return_recv_hook=True)
        self.assertEqual(result[:6],('x','ids','weights',8,896,1))
        self.assertTrue(result[6]['return_recv_hook'])
    def test_native_keyword_int8(self):
        result=self.dispatch(None,'x','ids',topk_weight='weights',num_max_dispatch_tokens_per_rank=8,num_experts=896,quant_type=1)
        self.assertEqual(result[:6],('x','ids','weights',8,896,1))
    def test_old_fp8(self):
        result=self.dispatch(None,'x','ids',8,896,topk_weights='weights',use_fp8=True)
        self.assertEqual(result[:6],('x','ids','weights',8,896,2))
        self.assertEqual(result[6]['quant_group_size'],128)
    def test_old_bf16(self):
        result=self.dispatch(None,'x','ids',8,896,topk_weights='weights',use_fp8=False)
        self.assertEqual(result[5],0)
    def test_native_signature_visible(self):
        self.assertIn('quant_type',inspect.signature(self.dispatch).parameters)
    def test_old_caller_frame_weights(self):
        topk_weights=self.Tensor()
        result=self.dispatch(None,'x','ids',8,896,use_fp8=True)
        self.assertIs(result[2],topk_weights)

if __name__=='__main__':unittest.main(verbosity=2)
