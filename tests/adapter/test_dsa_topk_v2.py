import torch
from hcu_adapter.dsa_topk_v2 import _wrap_paged_v2


def test_deterministic_and_raw_indices_survive_stride_padding():
    seen = []
    marker = object()
    def original(*args, **kwargs):
        seen.append((args, kwargs))
        return marker
    scores = torch.arange(10).reshape(2, 5).float()
    raw = torch.empty((2, 2), dtype=torch.int32)
    result = _wrap_paged_v2(original)(scores, 'lens', 'pages', 'out', 64,
                                     'plan', raw, deterministic=True)
    args, kwargs = seen[0]
    assert result is marker
    assert kwargs == {'deterministic': True}
    assert args[-1] is raw
    assert args[0].shape == (2, 8)
    torch.testing.assert_close(args[0][:, :5], scores)
    assert torch.isneginf(args[0][:, 5:]).all()


def test_aligned_scores_keep_identity_and_legacy_calls_receive_no_new_keyword():
    seen = []
    def legacy(scores, lens, pages, out, size, plan, raw):
        seen.append(scores)
    scores = torch.zeros((2, 8))
    _wrap_paged_v2(legacy)(scores, None, None, None, 64, None)
    assert seen[0] is scores
