import ast
import functools
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]/'hcu_adapter'


def function(file,name,namespace):
    tree=ast.parse((ROOT/file).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),namespace)
    return namespace[name]


class StartupCompatibility(unittest.TestCase):
    def test_cuda_version_gate_old_new_and_aliases(self):
        def check(*args,**kwargs):raise AssertionError('hardware version queried')
        ns={'_check_cuda_device_version':check}
        exec('def current():\n return _check_cuda_device_version([9],(12,3))\n',ns)
        current=functools.lru_cache(1)(ns['current'])
        legacy=functools.lru_cache(1)(functools.partial(check,[9],(12,3)))
        common=types.SimpleNamespace(_check_cuda_device_version=check,current=current,alias=current,legacy=legacy,untouched=lambda:True)
        pkg=types.SimpleNamespace()
        method=function('platform_seam.py','_replace_version_gates',{})
        names=method(common,pkg)
        self.assertEqual(set(names),{'current','alias','legacy'})
        for name in names:
            self.assertIs(getattr(common,name)(),False)
            self.assertIs(getattr(pkg,name)(),False)
        self.assertTrue(common.untouched())

    def test_spawn_device_resolves_new_and_legacy_module_locations(self):
        for current in [True,False]:
            with self.subTest(current=current):
                modules={name:types.ModuleType(name) for name in ['sglang','sglang.srt','sglang.srt.utils','sglang.srt.utils.common','sglang.srt.utils.device','hcu_adapter']}
                for module in modules.values():module.__path__=[]
                old,new=object(),object()
                modules['sglang.srt.utils.common'].maybe_reindex_device_id=old
                modules['sglang.srt.utils.device'].maybe_reindex_device_id=new
                if not current:modules['sglang.srt.utils.device']=None
                calls=[];registered=[]
                modules['hcu_adapter'].registry=types.SimpleNamespace(MECH_REBIND='rebind',register=lambda *a,**kw:registered.append(kw))
                method=function('sglang_spawn_device.py','install',{'_install_function':calls.append,'_device_context':object(),'LOG':types.SimpleNamespace(warning=lambda *a:None)})
                with patch.dict(sys.modules,modules):method()
                self.assertEqual(calls,[new if current else old])
                expected='sglang.srt.utils.device:maybe_reindex_device_id' if current else 'sglang.srt.utils.common:maybe_reindex_device_id'
                self.assertEqual(registered[0]['target'],expected)


if __name__=='__main__':unittest.main(verbosity=2)
