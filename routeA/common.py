import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _strip(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_backbone(ckpt_path, encoder_model=None, device='cuda'):
    from models.network import encoder_dict
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location='cpu')
    meta = {}
    if isinstance(ck, dict) and 'backbone' in ck:
        sd = ck['backbone']
    elif isinstance(ck, dict) and 'model' in ck:
        sd = ck['model']
    elif isinstance(ck, dict) and 'state_dict' in ck:
        sd = ck['state_dict']
    else:
        sd = ck
    if isinstance(ck, dict):
        for k in ('encoder_model', 'ndcn_cfg', 'routeA_cfg', 'routeB_cfg', 'epoch_index', 'best_prec1',
                  'data_name', 'shot_num'):
            if k in ck:
                meta[k] = ck[k]
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    if any(k.startswith('features.') for k in sd):
        sd = _strip(sd, 'features.')
    enc = encoder_model or meta.get('encoder_model')
    if enc is None:
        raise ValueError('%s does not record encoder_model; pass --encoder_model' % ckpt_path)
    if meta.get('encoder_model') not in (None, enc):
        raise ValueError('checkpoint is %s but --encoder_model=%s' % (meta['encoder_model'], enc))
    net = encoder_dict[enc]()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError('backbone keys do not match (missing %d, unexpected %d): %s'
                         % (len(missing), len(unexpected), (list(missing) + list(unexpected))[:5]))
    meta['encoder_model'] = enc
    return net.to(device).eval(), meta


PAPER_N_ITER = 1


def ndm_settings(meta, rank=None):
    cfg = meta.get('ndcn_cfg') or meta.get('routeA_cfg') or meta.get('routeB_cfg') or {}
    r = rank if rank is not None else int(cfg.get('rank', 5))
    recorded = cfg.get('n_iter')
    return r, PAPER_N_ITER, (int(recorded) if recorded is not None else None)


def file_sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(n), b''):
            h.update(chunk)
    return h.hexdigest()


def read_split(dataset_dir, split):
    rows = []
    with open(os.path.join(dataset_dir, split + '.csv')) as f:
        r = csv.reader(f)
        next(r)
        for line in r:
            if line:
                rows.append((line[0], line[1]))
    classes = sorted({c for _, c in rows})
    by_cls = {c: [] for c in classes}
    for i, (_, c) in enumerate(rows):
        by_cls[c].append(i)
    return rows, classes, by_cls


def eval_transform(image_size=84):
    import torchvision.transforms as T
    return T.Compose([T.Resize(92 if image_size == 84 else 256), T.CenterCrop(image_size),
                      T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class _Images(torch.utils.data.Dataset):
    def __init__(self, root, names, tf, occlude=0.0, occ_seed=0):
        self.root, self.names, self.tf = root, names, tf
        self.occlude, self.occ_seed = occlude, occ_seed

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(os.path.join(self.root, 'images', self.names[i])) as im:
            x = self.tf(im.convert('RGB'))
        if self.occlude > 0:
            _, H, W = x.shape
            side = int(round((self.occlude ** 0.5) * min(H, W)))
            rng = np.random.RandomState(self.occ_seed * 1000003 + i)
            y0, x0 = rng.randint(0, H - side + 1), rng.randint(0, W - side + 1)
            x = x.clone()
            x[:, y0:y0 + side, x0:x0 + side] = 0.0
        return x


@torch.no_grad()
def extract_split(net, dataset_dir, split, ndm, keep_maps=True, batch=128, workers=8,
                  device='cuda', image_size=84, occlude=0.0):
    from routeA.disc_pool import flatten, image_components
    rows, classes, by_cls = read_split(dataset_dir, split)
    ds = _Images(dataset_dir, [r[0] for r in rows], eval_transform(image_size), occlude)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers,
                                     shuffle=False, pin_memory=(device == 'cuda'))
    G, EZ, M, S, Zs = [], [], [], [], []
    hw, neg = None, 0.0
    for x in dl:
        f = net(x.to(device, non_blocking=True)).float()
        hw = tuple(f.shape[2:])
        neg = max(neg, float((f < 0).float().mean()))
        Z = torch.relu(flatten(f))
        if ndm is None:
            G.append(Z.mean(1).cpu())
        else:
            c = image_components(Z, 'ndm', ndm)
            G.append(c['g'].cpu()); EZ.append(c['ez'].cpu()); M.append(c['m'].cpu())
            S.append(c['s'].cpu())
        if keep_maps:
            Zs.append(Z.half().cpu())
    out = {'g': torch.cat(G), 'hw': hw, 'L': hw[0] * hw[1], 'names': [r[0] for r in rows],
           'labels': [r[1] for r in rows], 'classes': classes,
           'by_cls': by_cls, 'neg_fraction_max': neg}
    if EZ:
        out.update(ez=torch.cat(EZ), m=torch.cat(M), s=torch.cat(S))
    if keep_maps:
        out['Z'] = torch.cat(Zs)
    return out


def sample_episodes(by_cls, classes, n_episodes, way, shot, query, seed):
    rng = np.random.RandomState(seed)
    eps = np.empty((n_episodes, way, shot + query), dtype=np.int64)
    for e in range(n_episodes):
        cls = rng.choice(len(classes), way, replace=False)
        for j, c in enumerate(cls):
            pool = by_cls[classes[c]]
            if len(pool) < shot + query:
                raise ValueError('class %s has only %d images' % (classes[c], len(pool)))
            eps[e, j] = rng.choice(pool, shot + query, replace=False)
    return eps


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
