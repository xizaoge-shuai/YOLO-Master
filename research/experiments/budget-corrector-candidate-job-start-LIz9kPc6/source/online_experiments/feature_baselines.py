#!/usr/bin/env python3
"""FroFA/LOFF-TA idea adaptations for the existing D1 detection pilot.

Not a reproduction of either paper's complete architecture, recipe or scores.
FroFA: per-image/per-channel range, additive brightness, clipping, inverse map.
LOFF-TA-style: feature-space horizontal flip/center zoom plus Gaussian noise.
The original detector, optimizer and unaugmented validation path are retained.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'cache1': dict(views=1, geom=False, brightness=0., noise=0.),
    'cache2': dict(views=2, geom=False, brightness=0., noise=0.),
    'frofa-c1-d01': dict(views=1, geom=False, brightness=.1, noise=0.),
    'frofa-c1-d03': dict(views=1, geom=False, brightness=.3, noise=0.),
    'frofa-c2-d01': dict(views=2, geom=False, brightness=.1, noise=0.),
    'frofa-c2-d03': dict(views=2, geom=False, brightness=.3, noise=0.),
    'transport-c1': dict(views=1, geom=True, brightness=0., noise=0.),
    'loffta-c1-n005': dict(views=1, geom=True, brightness=0., noise=.05),
    'transport-c2': dict(views=2, geom=True, brightness=0., noise=0.),
    'loffta-c2-n005': dict(views=2, geom=True, brightness=0., noise=.05),
}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frofa(x, generator, magnitude):
    import torch
    x = x.float()
    if magnitude == 0:
        return x
    lo, hi = x.amin((-2, -1), keepdim=True), x.amax((-2, -1), keepdim=True)
    span = hi - lo
    delta = (torch.rand((*x.shape[:2], 1, 1), device=x.device, generator=generator) * 2 - 1) * magnitude
    z = ((x-lo)/span.clamp_min(1e-8) + delta).clamp(0, 1)
    return z*span + lo


def warp_features(x, flips, scales):
    import torch
    import torch.nn.functional as F
    x = x.float()
    if len(flips) != len(x) or len(scales) != len(x) or min(scales) <= 0:
        raise ValueError('Invalid feature geometry')
    if all(s == 1. for s in scales):
        mask = torch.tensor(flips, device=x.device).view(-1, 1, 1, 1)
        return torch.where(mask, x.flip(-1), x)
    theta = x.new_zeros((len(x), 2, 3))
    scale = x.new_tensor(scales)
    theta[:, 0, 0] = x.new_tensor([-1. if f else 1. for f in flips])/scale
    theta[:, 1, 1] = 1/scale
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    # Raw DINO features are neither RGB nor constrained to [0,1].
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def transform_targets(classes, boxes, flip, scale, imgsz):
    import torch
    if scale <= 0:
        raise ValueError('scale must be positive')
    boxes = boxes.clone()
    if scale != 1. and boxes.numel():
        lo = (boxes[:, :2] - boxes[:, 2:]/2 - .5)*scale + .5
        hi = (boxes[:, :2] + boxes[:, 2:]/2 - .5)*scale + .5
        before = (hi-lo).prod(dim=1)
        lo, hi = lo.clamp(0, 1), hi.clamp(0, 1)
        size = hi-lo
        keep = (size*imgsz >= 1).all(dim=1)
        keep &= size.prod(dim=1)/before.clamp_min(1e-12) >= .1
        boxes = torch.cat(((hi+lo)/2, size), dim=1)[keep]
        classes = classes[keep]
    if flip:
        boxes[:, 0] = 1-boxes[:, 0]
    return classes, boxes


def add_noise(x, generator, magnitude):
    import torch
    if magnitude == 0:
        return x
    # Detection adaptation: normalize noise strength to each channel's spatial std.
    sigma = x.std((-2, -1), correction=0, keepdim=True)
    return x + torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)*sigma*magnitude


class Augmenter:
    def __init__(self, method, seed, imgsz, device, scale_min, scale_max):
        import torch
        self.spec = CASES[method]
        self.imgsz, self.scale_min, self.scale_max = imgsz, scale_min, scale_max
        self.rng = torch.Generator(device=device).manual_seed(seed + 32452843)
        self.samples = 0

    def make_batch(self, banks, ids, collate, geom_rng):
        items, flips, scales = [], [], []
        spec = self.spec
        for i in ids:
            flip = geom_rng.random() < .5 if spec['geom'] or spec['views'] == 2 else False
            scale = geom_rng.uniform(self.scale_min, self.scale_max) if spec['geom'] else 1.
            view = int(flip) if spec['views'] == 2 else 0
            item = dict(banks[view][i])
            feature_flip = bool(flip and spec['views'] == 1)
            if spec['geom']:
                item['cls'], item['bboxes'] = transform_targets(
                    item['cls'], item['bboxes'], feature_flip, scale, self.imgsz)
            items.append(item)
            flips.append(feature_flip)
            scales.append(scale)
        batch = collate(items)
        if spec['geom'] or spec['brightness'] or spec['noise']:
            levels = []
            for x in batch['features']:
                x = x.float()
                if spec['geom']:
                    x = warp_features(x, flips, scales)
                if spec['brightness']:
                    x = frofa(x, self.rng, spec['brightness'])
                if spec['noise']:
                    x = add_noise(x, self.rng, spec['noise'])
                levels.append(x)
            batch['features'] = tuple(levels)
        self.samples += len(ids)
        return batch


BATCHES = '''
class Batches:
    def __init__(self, training):
        self.training = training
    def __iter__(self):
        ds = train if self.training else val
        indices = torch.randperm(len(ds), generator=generator).tolist() if self.training else list(range(len(ds)))
        for start in range(0, len(indices), a.batch):
            ids = indices[start:start+a.batch]
            if self.training:
                yield make_training_batch(train_bank, ids, d.collate_cached, aug_rng)
            else:
                yield d.collate_cached([val_bank[i] for i in ids])
'''


def patched(source):
    from online_experiments.flip_cache_probe import patched as cached_patch
    tree = cached_patch(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    hits = 0
    for i, node in enumerate(main.body):
        if isinstance(node, ast.ClassDef) and node.name == 'Batches':
            main.body[i] = ast.parse(BATCHES).body[0]
            hits += 1
    if hits != 1:
        raise RuntimeError('Pilot batching layout changed')
    return ast.fix_missing_locations(tree)


def worker(a):
    import torch
    from scripts import d1_train_cached_detector as d
    from online_experiments import d1_online_compare as pilot
    from online_experiments import flip_cache_probe as f
    from online_experiments.fusion_probe import configure_fusion

    torch.set_num_threads(a.threads)
    provenance = read(a.job/'provenance.json')
    f.verify(provenance['sources'])
    source = Path(provenance['source_job'])
    cache_root = source/'view-cache'
    if sha(cache_root/'manifest.json') != provenance['view_manifest_sha256']:
        raise RuntimeError('View cache manifest changed')
    meta = read(cache_root/'manifest.json')
    f.verify(meta['sources'])
    f.verify(meta['inputs'])
    if torch.__version__ != meta['torch_version']:
        raise RuntimeError('Torch version differs from existing feature extraction')
    cfg = read(a.job/'reference_args.json')
    spec = CASES[a.method]
    cfg.update(mode='offline', augment='none', parity_only=False, seed=a.seed,
               epochs=a.epochs, device='cuda:0', output=a.job/a.phase/f'{a.method}-s{a.seed}')
    cfg['cache'] = Path(cfg['cache'])
    aug = Augmenter(a.method, a.seed, int(meta['identity']['imgsz']), 'cuda:0',
                    cfg['scale_min'], cfg['scale_max'])

    def cached_bank(dataset, device):
        if [s['sample_id'] for s in dataset.samples] != meta['split']['train_sample_ids']:
            raise RuntimeError('Training cache sample order changed')
        banks = []
        for view in range(spec['views']):
            bank = []
            for i, sample in enumerate(dataset.samples):
                rel = f'v{view}/{i:06d}.pt'
                payload = (cache_root/rel).read_bytes()
                if hashlib.sha256(payload).hexdigest() != meta['files'][rel]:
                    raise RuntimeError(f'Corrupt view-cache file: {rel}')
                item = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
                if item['sample_id'] != sample['sample_id'] or item['image_path'] != sample['image_path']:
                    raise RuntimeError(f'Cache identity mismatch: {rel}')
                item['features'] = tuple(x.to(device) for x in item['features'])
                bank.append(item)
            banks.append(bank)
        return banks

    base, original_write = d.CachedLatentDetector, d.atomic_write_json
    class MeanDetector(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            configure_fusion(self, 'mean_value')

    def write(path, value):
        name = Path(path).name
        if name == 'split.json' and any(value[k] != meta['split'][k] for k in meta['split']):
            raise RuntimeError('Train/validation split changed')
        if name in ('args.json', 'summary.json'):
            value = dict(value)
            cost, size = f.cache_cost(meta, spec['views'])
            value.update(method=a.method, detector_fusion='mean_value',
                feature_augmentation=spec, training_teacher_calls=0, training_teacher_images=0,
                actual_new_cache_build_seconds=0., charged_training_cache_build_seconds=cost,
                training_cache_bytes=size, existing_validation_cache_build_included=False,
                adaptation='FroFA/LOFF-TA ideas adapted to detection; not full paper reproduction')
            if name == 'args.json':
                if value['cache_identity'] != meta['identity']:
                    raise RuntimeError('Cache identity differs from prior experiment')
                value.update(source_hashes=provenance['sources'], cpu_threads=a.threads,
                    view_cache=str(cache_root), visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    feature_geometry='bilinear center zoom; zero feature padding; no feature clipping',
                    brightness_policy='per-image/channel minmax, U(-delta,delta), clip [0,1], invert',
                    noise_policy='sigma = magnitude * spatial population std per image/channel after geometry',
                    scale_interval=[cfg['scale_min'],cfg['scale_max']],
                    rng_policy='isolated torch feature RNG; existing pilot shuffle/geometry RNG',
                    validation_policy='unchanged original validation cache; no feature augmentation')
            else:
                expected = value['completed_epochs'] * cfg['train_count']
                if aug.samples != expected:
                    raise RuntimeError(f'Unexpected sample count {aug.samples} != {expected}')
                value.update(training_samples=aug.samples, build_plus_run_seconds=cost+value['wall_seconds'])
        return original_write(path, value)

    namespace = dict(vars(pilot))
    exec(compile(patched(Path(pilot.__file__).read_text()), pilot.__file__, 'exec'), namespace)
    namespace.update(arguments=lambda: argparse.Namespace(**cfg), cached_bank=cached_bank,
                     make_training_batch=aug.make_batch)
    d.CachedLatentDetector, d.atomic_write_json = MeanDetector, write
    try:
        namespace['main']()
    finally:
        d.CachedLatentDetector, d.atomic_write_json = base, original_write


def checked_result(folder, epochs):
    summary, args = read(folder/'summary.json'), read(folder/'args.json')
    with (folder/'results.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if summary['completed_epochs'] != epochs or [int(r['epoch']) for r in rows] != list(range(1, epochs+1)):
        raise RuntimeError(f'Incomplete results: {folder}')
    if any(not math.isfinite(float(v)) for r in rows for v in r.values()):
        raise RuntimeError(f'Nonfinite results: {folder}')
    if args['detector_fusion'] != 'mean_value':
        raise RuntimeError(f'Incorrect head: {folder}')
    return summary, args, rows


def report(job, epochs):
    records = []
    for method in CASES:
        for seed in (0, 1, 2):
            folder = job/'train'/f'{method}-s{seed}'
            if not (folder/'DONE').exists():
                continue
            s, _, rows = checked_result(folder, epochs)
            records.append(dict(method=method,seed=seed,best_AP=100*s['best_map50_95'],
                last_AP=100*s['final_map50_95'],best_epoch=int(max(rows,key=lambda r:float(r['map50_95']))['epoch']),
                train_s=s['train_seconds'],wall_s=s['wall_seconds'],
                charged_build_s=s['charged_training_cache_build_seconds'],
                build_plus_wall_s=s['build_plus_run_seconds'],
                cache_GiB=s['training_cache_bytes']/2**30,VRAM_GiB=s['peak_vram_gib']))
    lines = ['AP: 0-100. Detection adaptations of paper ideas, not original paper reproductions.',
             'All new runs: same mean_value head, original validation cache, serial GPU0.',
             'Existing train cache reused. build+wall charges archived train-cache build once per run.',
             'Validation-cache build excluded. No backbone calls during these training runs.',
             'method seed best_AP last_AP best_epoch train_s wall_s build+wall_s cache_GiB VRAM_GiB']
    for r in records:
        lines.append(f"{r['method']} {r['seed']} {r['best_AP']:.4f} {r['last_AP']:.4f} {r['best_epoch']} "
                     f"{r['train_s']:.1f} {r['wall_s']:.1f} {r['build_plus_wall_s']:.1f} {r['cache_GiB']:.3f} {r['VRAM_GiB']:.3f}")
    lines += ['', 'method n best_AP_mean sample_sd last_AP_mean train_s_mean build+wall_s_mean']
    for method in CASES:
        group = [r for r in records if r['method'] == method]
        if not group:
            continue
        mean = statistics.mean
        sd = f"{statistics.stdev(r['best_AP'] for r in group):.4f}" if len(group)>1 else 'NA'
        lines.append(f"{method} {len(group)} {mean(r['best_AP'] for r in group):.4f} {sd} "
                     f"{mean(r['last_AP'] for r in group):.4f} {mean(r['train_s'] for r in group):.1f} "
                     f"{mean(r['build_plus_wall_s'] for r in group):.1f}")
    dump(job/'comparison.json', records)
    text = '\n'.join(lines)+'\n'
    temp = job/'comparison.txt.tmp'
    temp.write_text(text)
    temp.replace(job/'comparison.txt')
    return text


def campaign(a):
    import fcntl
    from online_experiments import flip_cache_probe as f
    lock_path = ROOT/'runs/paper/feature-baselines/.campaign.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    campaign_lock = lock_path.open('a')
    try:
        fcntl.flock(campaign_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another feature-baselines campaign is already running') from None
    source = a.source.resolve()
    if (source/'status.txt').read_text().strip() != 'SUCCEEDED':
        raise RuntimeError('flip-cache reference must be SUCCEEDED')
    if (a.job/'plan.json').exists():
        raise RuntimeError('Use a fresh job directory; previous runs are never overwritten')
    meta = read(source/'view-cache/manifest.json')
    f.verify(meta['sources'])
    f.verify(meta['inputs'])
    plan = read(source/'plan.json')
    cfg = read(source/'reference_args.json')
    ref = read(source/'train/cache1-s0/args.json')
    a.epochs, a.threads = plan['epochs'], plan['threads']
    common = ('cache_identity','epochs','batch','lr','weight_decay','aux_weight',
              'teacher_dtype','train_count','val_count','split_seed','detector_fusion','cpu_threads')
    for views in (1,2):
        for seed in (0,1,2):
            folder = source/'train'/f'cache{views}-s{seed}'
            if not (folder/'DONE').exists():
                raise RuntimeError(f'Incomplete source: {folder}')
            _, c, _ = checked_result(folder,a.epochs)
            f.verify(c['source_hashes'])
            split = read(folder/'split.json')
            if any(c[k] != ref[k] for k in common) or any(split[k] != meta['split'][k] for k in meta['split']):
                raise RuntimeError(f'Inconsistent source settings: {folder}')
    cfg_keys = ('cache_identity','batch','lr','weight_decay','aux_weight','teacher_dtype',
                'train_count','val_count','split_seed')
    if any(cfg[k] != ref[k] for k in cfg_keys) or ref['cpu_threads'] != a.threads:
        raise RuntimeError('Snapshot does not match completed reference')
    if not 0 < cfg['scale_min'] <= cfg['scale_max']:
        raise RuntimeError('Invalid scale interval')
    sources = dict(meta['sources'])
    sources['online_experiments/feature_baselines.py'] = sha(Path(__file__))
    provenance = dict(source_job=str(source),sources=sources,
        view_manifest_sha256=sha(source/'view-cache/manifest.json'),
        sources_url=['https://arxiv.org/abs/2403.10519','https://arxiv.org/abs/2410.02527'])
    dump(a.job/'provenance.json',provenance)
    dump(a.job/'reference_args.json',cfg)
    dump(a.job/'reference_split.json',meta['split'])
    patched((ROOT/'online_experiments/d1_online_compare.py').read_text())
    uuid = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if not uuid.startswith('GPU-') or len(uuid.splitlines()) != 1:
        raise RuntimeError('Cannot resolve GPU0 UUID')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads))
    tasks=[]
    for seed in (0,1,2):
        methods=list(CASES)
        random.Random(49999+seed).shuffle(methods)
        tasks.extend((m,seed) for m in methods)
    dump(a.job/'plan.json',dict(epochs=a.epochs,threads=a.threads,gpu_uuid=uuid,
         cases=CASES,tasks=tasks,source_job=str(source),schema_version=1,
         note='Fixed augmentation strengths, all reported; validation pilot is not held-out test performance'))
    (a.job/'status.txt').write_text('RUNNING GPU0 preflight\n')
    with (a.job/'preflight.log').open('w') as stream:
        result=subprocess.run([sys.executable,'-c',
            "import torch; assert torch.cuda.is_available(); x=torch.randn(256,256,device='cuda'); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print(torch.cuda.get_device_name(0))"],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('GPU0 preflight failed; inspect preflight.log')

    def run(method,seed,phase,number,total):
        label=f'{phase}-{method}-s{seed}'
        (a.job/'status.txt').write_text(f'RUNNING GPU0 {label} ({number}/{total})\n')
        epochs=1 if phase=='smoke' else a.epochs
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(a.job),
                 '--stage','worker','--method',method,'--seed',str(seed),'--phase',phase,
                 '--epochs',str(epochs),'--threads',str(a.threads)]
        log=a.job/f'{label}.log'
        print(f'START {label} ({number}/{total}) log={log}',flush=True)
        started=time.perf_counter()
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-60:]),flush=True)
            raise RuntimeError(f'Stopped on failure: {log}')
        folder=a.job/phase/f'{method}-s{seed}'
        s,c,_=checked_result(folder,epochs)
        if s['method'] != method or s['seed'] != seed or s['training_teacher_calls'] != 0:
            raise RuntimeError(f'Unexpected worker metadata: {folder}')
        if c['source_hashes'] != sources:
            raise RuntimeError('Worker source identity changed')
        (folder/'DONE').write_text('exit=0\n')
        print(f'END {label} seconds={time.perf_counter()-started:.1f}',flush=True)
    for i,method in enumerate(CASES,1):
        run(method,0,'smoke',i,len(CASES))
    for i,(method,seed) in enumerate(tasks,1):
        run(method,seed,'train',i,len(tasks))
        report(a.job,a.epochs)
    f.verify(sources)
    f.verify(meta['inputs'])
    (a.job/'status.txt').write_text('SUCCEEDED\n')
    print(report(a.job,a.epochs),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--stage',choices=['run','worker'],default='run')
    p.add_argument('--method',choices=list(CASES),default='cache1')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--phase',choices=['smoke','train'],default='train')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    a.job=a.job.resolve()
    a.job.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT))
    os.chdir(ROOT)
    try:
        if a.stage=='run':
            if a.source is None:
                raise ValueError('--source is required for a campaign')
            campaign(a)
        else:
            worker(a)
    except BaseException as exc:
        if a.stage=='run':
            (a.job/'status.txt').write_text(f'FAILED: {exc}\n')
        raise


if __name__=='__main__':
    main()
