import ast
import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
path = HERE / "flip_cache_probe.py"
if path.exists():
    spec = importlib.util.spec_from_file_location("flip_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
else:
    probe = SimpleNamespace()

def test_teacher_and_cached_flip_draws_match():
    assert hasattr(probe, "select_view")
    for seed in range(3):
        actual = random.Random(seed + 104729)
        expected = random.Random(seed + 104729)
        assert [probe.select_view(actual, 2) for _ in range(1200)] == [
            int(expected.random() < .5) for _ in range(1200)]
    rng = random.Random(11)
    before = rng.getstate()
    assert probe.select_view(rng, 1) == 0
    assert rng.getstate() == before

def test_pilot_transform_is_checked():
    assert hasattr(probe, "patched")
    source = (HERE / "d1_online_compare.py").read_text()
    compile(probe.patched(source), "<patched>", "exec")
    with pytest.raises(RuntimeError):
        probe.patched(source.replace("train_bank = bank(train)", "other = bank(train)"))

def test_cached_batches_preserve_feature_label_pairs_and_validation():
    assert hasattr(probe, "patched")
    import torch
    tree = probe.patched((HERE / "d1_online_compare.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Batches")
    original = [dict(sample_id=i, features=("original", i), bboxes=.2) for i in range(8)]
    flipped = [dict(sample_id=i, features=("flipped", i), bboxes=.8) for i in range(8)]
    ns = dict(torch=torch, train=range(8), val=range(8), train_bank=[original, flipped],
              val_bank=original, generator=torch.Generator().manual_seed(0),
              aug_rng=random.Random(104729), select_view=probe.select_view,
              a=SimpleNamespace(batch=3, views=2), d=SimpleNamespace(collate_cached=lambda x:x))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, "<batches>", "exec"), ns)
    items = [x for b in ns["Batches"](True) for x in b]
    assert sorted(x["sample_id"] for x in items) == list(range(8))
    expected = random.Random(104729)
    for item in items:
        flip = expected.random() < .5
        assert item["features"] == ("flipped" if flip else "original", item["sample_id"])
        assert item["bboxes"] == (.8 if flip else .2)
    before = ns["aug_rng"].getstate()
    assert [x for b in ns["Batches"](False) for x in b] == original
    assert ns["aug_rng"].getstate() == before

def test_cache_cost_counts_only_selected_views_plus_shared_setup():
    assert hasattr(probe, "cache_cost")
    meta = dict(setup_seconds=3, view_seconds=[10,12], view_bytes=[100,120])
    assert probe.cache_cost(meta, 1) == (13,100)
    assert probe.cache_cost(meta, 2) == (25,220)

def test_builder_extracts_flipped_images_not_flipped_features(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import torch
    assert hasattr(probe, 'build')
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    original_verify = probe.verify
    monkeypatch.setattr(probe, 'verify', lambda hashes: original_verify(hashes, tmp_path))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(probe.shutil, 'disk_usage', lambda p: SimpleNamespace(free=10**12))
    job, cache, dataset = tmp_path/'job', tmp_path/'original-cache', tmp_path/'dataset'
    for p in (job, cache, dataset):
        p.mkdir()
    base_image = torch.tensor([[[0.,1.],[.25,.75]]]).repeat(3,1,1)
    position = torch.tensor([[[0.,.125],[0.,.125]]])
    layers = [3,7,11]
    samples = []
    for i in range(3):
        image = dataset/f'{i}.bin'
        image.write_bytes(bytes([i]))
        image.with_suffix('.txt').write_text('0 .25 .5 .2 .2')
        samples.append(dict(sample_id=str(i), image_path=image.name, image_sha256=probe.sha(image),
                            letterbox={}, shapes={f'layer_{l}':[1,2,2] for l in layers}))
    ident = dict(model_id='fake', revision='fixed', layers=layers, imgsz=2)
    (cache/'manifest.json').write_text(json.dumps(dict(identity=ident, samples=samples)))
    (job/'reference_args.json').write_text(json.dumps(dict(cache=str(cache),cache_identity=ident,
        dataset=str(dataset),train_count=2,val_count=1,split_seed=0,batch=2,teacher_dtype='fp32',parity_tolerance=.02)))
    (job/'reference_split.json').write_text(json.dumps(dict(train_sample_ids=['0','1'],val_sample_ids=['2'])))
    (job/'baseline_sources.json').write_text('{}')
    for rel in ('online_experiments/flip_cache_probe.py','ultralytics/data/foundation_cache.py',
                'ultralytics/nn/foundation/preprocessing.py','ultralytics/utils/loss.py'):
        p=tmp_path/rel
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('source')
    class Dataset:
        def __init__(self,*args):
            self.samples=samples[:2]
            self.dataset_root=dataset
        def __len__(self):
            return 2
        def __getitem__(self,i):
            return {'features':tuple((base_image[:1]+position).half() for _ in layers)}
    d=SimpleNamespace(validate_manifest=lambda *a,**k:None,check_det_dataset=lambda *a,**k:{'names':['x']},
        split_sample_indices=lambda *a:([0,1],[2]),CachedDetectionDataset=Dataset,
        label_path=lambda root,path:(root/path).with_suffix('.txt'),seed_everything=lambda seed:None,
        read_yolo_labels=lambda *a:(torch.tensor([[0.]]),torch.tensor([[.25,.5,.2,.2]])))
    scripts=types.ModuleType('scripts')
    scripts.d1_train_cached_detector=d
    monkeypatch.setitem(sys.modules,'scripts',scripts)
    calls=[]
    class Extractor:
        def __init__(self,*a,**k):
            pass
        def __call__(self,images):
            calls.append(images.clone())
            return {f'layer_{l}':images[:,:1]+position for l in layers}
    extractor_module=types.ModuleType('scripts.d1_build_feature_cache')
    extractor_module.DINOv3MultiLevelExtractor=Extractor
    monkeypatch.setitem(sys.modules,'scripts.d1_build_feature_cache',extractor_module)
    foundation=types.ModuleType('ultralytics.data.foundation_cache')
    foundation.load_letterboxed_tensor=lambda *a,**k:(base_image.clone(),{})
    monkeypatch.setitem(sys.modules,'ultralytics.data.foundation_cache',foundation)
    probe.build(SimpleNamespace(job=job,threads=1))
    original=torch.load(job/'view-cache/v0/000000.pt',weights_only=True)
    flipped=torch.load(job/'view-cache/v1/000000.pt',weights_only=True)
    assert torch.equal(calls[1],calls[0].flip(-1))
    assert torch.equal(flipped['features'][0],(base_image[:1].flip(-1)+position).half())
    assert not torch.equal(flipped['features'][0],original['features'][0].flip(-1))
    assert flipped['bboxes'][0,0].item()==.75
    assert original['bboxes'][0,0].item()==.25
    meta=probe.read(job/'view-cache/manifest.json')
    assert meta['teacher_calls']==2 and meta['teacher_images']==4
    assert len(meta['files'])==4 and max(meta['parity'])==0
