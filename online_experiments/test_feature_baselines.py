import ast
import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).parent
path = HERE / 'feature_baselines.py'
if path.exists():
    spec = importlib.util.spec_from_file_location('feature_baselines', path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
else:
    baseline = SimpleNamespace()


def test_frofa_channel_range_brightness_and_rng_isolation():
    assert hasattr(baseline, 'frofa')
    x = torch.tensor([[[[-4., 0.], [2., 8.]], [[7., 7.], [7., 7.]]]])
    saved = x.clone()
    torch.manual_seed(91)
    state = torch.random.get_rng_state().clone()
    actual = baseline.frofa(x, torch.Generator().manual_seed(5), .3)
    delta = (torch.rand((1, 2, 1, 1), generator=torch.Generator().manual_seed(5)) * 2 - 1) * .3
    lo, hi = x.amin((-2, -1), keepdim=True), x.amax((-2, -1), keepdim=True)
    expected = (((x-lo)/(hi-lo).clamp_min(1e-8)+delta).clamp(0, 1)*(hi-lo)+lo)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(x, saved)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(actual[:, 1], x[:, 1])
    assert torch.equal(baseline.frofa(x, torch.Generator(), 0), x)


def test_raw_feature_warp_does_not_clip_or_use_rgb_fill():
    assert hasattr(baseline, 'warp_features')
    x = torch.tensor([[[[-8., 3.], [7., 11.]]]])
    assert torch.equal(baseline.warp_features(x, [False], [1.]), x)
    assert torch.equal(baseline.warp_features(x, [True], [1.]), x.flip(-1))
    x = torch.full((1, 2, 8, 8), 9.)
    out = baseline.warp_features(x, [False], [.5])
    assert out[0, 0, 0, 0] == 0
    assert out.max() == 9


def test_box_geometry_matches_image_path_and_keeps_small_objects():
    assert hasattr(baseline, 'transform_targets')
    from online_experiments.d1_online_compare import geometry
    boxes = torch.tensor([[.4, .5, .005, .005], [.99, .4, .03, .03], [.2, .3, .1, .2]])
    cls = torch.tensor([[0.], [1.], [2.]])
    saved = boxes.clone()
    for flip in (False, True):
        for scale in (.85, 1., 1.15):
            _, ec, eb = geometry(torch.zeros(3, 640, 640), cls, boxes, flip, scale)
            ac, ab = baseline.transform_targets(cls, boxes, flip, scale, 640)
            assert torch.equal(ac, ec)
            torch.testing.assert_close(ab, eb)
            assert 0 in ac  # 3.2px is valid; using the 40-token grid would delete it.
    assert torch.equal(boxes, saved)
    ac, ab = baseline.transform_targets(cls[:0], boxes[:0], True, .9, 640)
    assert ac.shape == (0, 1) and ab.shape == (0, 4)


def collate(items):
    return dict(features=tuple(torch.stack([i['features'][level] for i in items]) for level in range(2)),
                targets=[(i['cls'], i['bboxes']) for i in items],
                sample_ids=[i['sample_id'] for i in items])


def test_c1_and_c2_share_spatial_schedule_and_never_mutate_cache():
    assert hasattr(baseline, 'Augmenter')
    banks = [[], []]
    for view in (0, 1):
        for i in range(4):
            # Use distinguishable real anchor values, not artificially flipped features.
            x = torch.arange(64.).reshape(1, 8, 8) + 100*view + i
            banks[view].append(dict(features=(x.half(), x.half()+1),
                cls=torch.tensor([[0.]]), bboxes=torch.tensor([[.2 if view == 0 else .8, .5, .1, .1]]), sample_id=i))
    saved = [[i['features'][0].clone() for i in bank] for bank in banks]
    r1, r2, reference = random.Random(12), random.Random(12), random.Random(12)
    a1 = baseline.Augmenter('transport-c1', 0, 640, 'cpu', .85, 1.15)
    a2 = baseline.Augmenter('transport-c2', 0, 640, 'cpu', .85, 1.15)
    b1 = a1.make_batch(banks[:1], [3, 1, 2], collate, r1)
    b2 = a2.make_batch(banks, [3, 1, 2], collate, r2)
    for j, i in enumerate([3, 1, 2]):
        flip, scale = reference.random() < .5, reference.uniform(.85, 1.15)
        x = banks[int(flip)][i]['features'][0][None].float()
        expected = baseline.warp_features(x, [False], [scale])[0]
        torch.testing.assert_close(b2['features'][0][j], expected)
        torch.testing.assert_close(b1['targets'][j][1], b2['targets'][j][1])
    assert r1.getstate() == r2.getstate() == reference.getstate()
    for view in (0, 1):
        for i in range(4):
            assert torch.equal(banks[view][i]['features'][0], saved[view][i])


def test_patching_leaves_validation_unaugmented():
    assert hasattr(baseline, 'patched')
    source = (HERE/'d1_online_compare.py').read_text()
    tree = baseline.patched(source)
    compile(tree, '<patched>', 'exec')
    with pytest.raises(RuntimeError):
        baseline.patched(source.replace('train_bank = bank(train)', 'other = bank(train)'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Batches')
    calls = []
    ns = dict(torch=torch, train=range(3), val=range(2), generator=torch.Generator(),
              a=SimpleNamespace(batch=2), train_bank=[['train']], val_bank=['v0', 'v1'],
              aug_rng=random.Random(1), d=SimpleNamespace(collate_cached=lambda x:x),
              make_training_batch=lambda *args: calls.append(args) or ['augmented'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<batches>', 'exec'), ns)
    assert list(ns['Batches'](False)) == [['v0', 'v1']]
    assert calls == []
    assert list(ns['Batches'](True)) == [['augmented'], ['augmented']]
    assert len(calls) == 2


def test_noise_is_reproducible_and_zero_noise_is_identity():
    assert hasattr(baseline, 'add_noise')
    x = torch.arange(96.).reshape(2, 3, 4, 4)
    assert torch.equal(baseline.add_noise(x, torch.Generator(), 0), x)
    y1 = baseline.add_noise(x, torch.Generator().manual_seed(9), .05)
    y2 = baseline.add_noise(x, torch.Generator().manual_seed(9), .05)
    assert torch.equal(y1, y2) and not torch.equal(y1, x)
    assert torch.isfinite(y1).all()
