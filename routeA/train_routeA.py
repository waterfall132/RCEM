import argparse
import json
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from routeA.common import load_backbone, extract_split, sample_episodes, read_split, PAPER_N_ITER
from routeA.disc_pool import flatten, image_components, route_a_logits, gap_logits, SCORES


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--data_name', required=True)
    p.add_argument('--encoder_model', default='ResNet12_ReLU',
                   help='Conv64F_Local_ReLU | ResNet12_ReLU (ReLU-terminated, as in the paper)')
    p.add_argument('--init', default='', help='checkpoint to initialise the backbone (optional)')
    p.add_argument('--outf', required=True)
    p.add_argument('--way', type=int, default=5)
    p.add_argument('--shot', type=int, default=1)
    p.add_argument('--query', type=int, default=15)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--episodes', type=int, default=4000, help='training episodes per epoch')
    p.add_argument('--val_episodes', type=int, default=600)
    p.add_argument('--val_seed', type=int, default=777)
    p.add_argument('--assign', default='ndm', choices=['ndm', 'point', 'smooth'])
    p.add_argument('--score', default='disc_lse', choices=[s for s in SCORES if s != 'random'])
    p.add_argument('--proto', default='gap', choices=['gap', 'loo'])
    p.add_argument('--T', type=float, default=0.1, help='component softmax temperature (from frozen val)')
    p.add_argument('--t', type=float, default=0.1, help='log-mean-exp temperature (disc_lse)')
    p.add_argument('--gamma', type=float, default=0.5, help='constant chosen on validation')
    p.add_argument('--rank', type=int, default=5)
    p.add_argument('--tau_init', type=float, default=10.0)
    p.add_argument('--optim', default='auto', choices=['auto', 'sgd', 'adam'])
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--clip_grad', type=float, default=5.0)
    p.add_argument('--dampening', type=float, default=0.0,
                   help='SGD dampening; DN4 official code uses 0.9')
    p.add_argument('--no_nesterov', action='store_true',
                   help='plain momentum SGD (DN4 official setting)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--continue_from', default='',
                   help='start from the full model state of this checkpoint (new optimizer and a new\n'
                        'cosine schedule over --epochs, i.e. a warm restart)')
    p.add_argument('--epoch_offset', type=int, default=0,
                   help='added to the epoch index when seeding the training episodes (no repeats)')
    p.add_argument('--no_resume', action='store_true',
                   help='ignore an existing model_last.pth.tar (default: resume an interrupted run)')
    return p.parse_args()


class RouteANet(nn.Module):
    def __init__(self, backbone, a):
        super().__init__()
        from models.ndcn import NDM
        self.features = backbone
        self.ndm = NDM(rank=a.rank, n_iter=PAPER_N_ITER)
        self.a = a
        self.raw_tau = nn.Parameter(torch.tensor(float(np.log(np.expm1(a.tau_init)))))

    @property
    def tau(self):
        return F.softplus(self.raw_tau)

    def comps(self, x):
        f = self.features(x)
        Z = torch.relu(flatten(f))
        return image_components(Z, self.a.assign, self.ndm, tuple(f.shape[2:]))

    def gap_only(self):
        return self.a.gamma == 0 and self.a.proto == 'gap'

    def forward(self, xq, xs, way, shot):
        if self.gap_only():
            g = lambda x: {'g': torch.relu(flatten(self.features(x))).mean(1)}
            return self.tau * gap_logits(g(xq), g(xs), way, shot)
        cq, cs = self.comps(xq), self.comps(xs)
        cos = route_a_logits(cq, cs, way, shot, self.a.score, [self.a.T], [self.a.gamma],
                             self.a.t, self.a.proto)[0, 0]
        return self.tau * cos


def loader_opt(a, epoch):
    return Namespace(dataset_dir=a.dataset_dir, way_num=a.way, shot_num=a.shot, query_num=a.query,
                     test_aug=False, aug_shot_num=20, episode_train_num=a.episodes,
                     episode_val_num=10, episode_test_num=10, train_aug=True, imageSize=84,
                     episodeSize=1, testepisodeSize=1, workers=a.workers, current_epoch=epoch)


@torch.no_grad()
def validate(model, a, episodes, dev):
    model.eval()
    feat = extract_split(model.features, a.dataset_dir, 'val', None if model.gap_only() else model.ndm,
                         keep_maps=(a.assign != 'ndm' and not model.gap_only()), workers=a.workers, device=dev)
    y = torch.arange(a.way, device=dev).repeat_interleave(a.query)
    accs = []
    for idx in episodes:
        s_idx, q_idx = idx[:, :a.shot].reshape(-1), idx[:, a.shot:].reshape(-1)
        def comp(ii):
            if model.gap_only():
                return {'g': feat['g'][ii].to(dev).float()}
            if a.assign == 'ndm':
                c = {k: feat[k][ii].to(dev).float() for k in ('g', 'ez', 'm', 's')}
                c['e'] = c['ez'] / (c['m'].unsqueeze(-1) + 1e-8)
                c['L'] = feat['L']
                return c
            return image_components(feat['Z'][ii].to(dev).float(), a.assign, None, feat['hw'])
        if model.gap_only():
            lg = gap_logits(comp(q_idx), comp(s_idx), a.way, a.shot)
        else:
            lg = route_a_logits(comp(q_idx), comp(s_idx), a.way, a.shot, a.score, [a.T],
                                [a.gamma], a.t, a.proto)[0, 0]
        accs.append((lg.argmax(1) == y).float().mean().item() * 100)
    model.train()
    accs = np.array(accs)
    return float(accs.mean()), float(1.96 * accs.std(ddof=1) / np.sqrt(len(accs)))


def main():
    a = parse()
    import dataset.general_dataloader as FewShotDataloader
    from models.network import encoder_dict
    os.makedirs(a.outf, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cpu':
        print('WARNING: no GPU, running on CPU (smoke test only)')
    if a.proto == 'loo' and a.shot < 2:
        raise ValueError('--proto loo needs --shot >= 2')

    if a.init:
        backbone, meta = load_backbone(a.init, a.encoder_model, 'cpu')
        print('backbone initialised from', a.init, meta.get('epoch_index', ''))
    else:
        backbone = encoder_dict[a.encoder_model]()
    model = RouteANet(backbone, a).to(dev)
    if a.continue_from:
        ck0 = torch.load(a.continue_from, map_location=dev, weights_only=False)
        model.load_state_dict(ck0['model'])
        print('continuing from %s (epoch %s, val %.2f)' % (a.continue_from, ck0.get('epoch_index'),
              float(ck0.get('best_prec1', float('nan')))), flush=True)
    opt_name = a.optim if a.optim != 'auto' else ('adam' if 'ResNet' in a.encoder_model else 'sgd')
    lr = a.lr if a.lr is not None else (1e-3 if opt_name == 'adam' else 0.05)
    if opt_name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=a.weight_decay)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=a.dampening, weight_decay=a.weight_decay,
                                nesterov=(not a.no_nesterov) and a.dampening == 0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, a.epochs, eta_min=lr * 1e-3)

    _, vclasses, vby = read_split(a.dataset_dir, 'val')
    val_eps = sample_episodes(vby, vclasses, a.val_episodes, a.way, a.shot, a.query, a.val_seed)
    cfg = {k: getattr(a, k) for k in ('assign', 'score', 'proto', 'T', 't', 'gamma', 'rank',
                                      'shot', 'way', 'seed')}
    cfg['n_iter'] = PAPER_N_ITER
    json.dump(dict(vars(a), n_iter=PAPER_N_ITER), open(os.path.join(a.outf, 'config.json'), 'w'), indent=2)
    log = open(os.path.join(a.outf, 'log.txt'), 'a')
    ce = nn.CrossEntropyLoss()
    best, hist, start = -1.0, [], 0
    last = os.path.join(a.outf, 'model_last.pth.tar')
    if not a.no_resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        if 'optim' in ck and 'sched' in ck:
            model.load_state_dict(ck['model']); optim.load_state_dict(ck['optim']); sched.load_state_dict(ck['sched'])
            start = int(ck['epoch_index']) + 1
            best = float(ck['best_prec1'])
            try:
                hist = json.load(open(os.path.join(a.outf, 'summary.json')))['history'][:start]
            except Exception:
                hist = []
            if 'rng' in ck:
                torch.set_rng_state(ck['rng']['torch'].cpu())
                if torch.cuda.is_available() and ck['rng'].get('cuda') is not None:
                    torch.cuda.set_rng_state_all([t.cpu() for t in ck['rng']['cuda']])
                np.random.set_state(ck['rng']['numpy'])
            print('RESUMED from epoch %d (best val so far %.2f)' % (start, best), flush=True)
        else:
            raise SystemExit('%s exists but has no optimizer state (old format): move the folder aside and '
                             'retrain, or pass --no_resume to overwrite' % last)
    for ep in range(start, a.epochs):
        random.seed(a.seed * 1000 + ep + a.epoch_offset)
        loader = FewShotDataloader.get_Fewshot_dataloader(loader_opt(a, ep + a.epoch_offset), ['train'])[0]
        model.train()
        t0, tot, cor, n = time.time(), 0.0, 0, 0
        for i, (qi, qt, si, st) in enumerate(loader):
            xq = torch.cat(qi, 0).to(dev, non_blocking=True)
            xs = torch.cat(si, 0).squeeze(0).to(dev, non_blocking=True)
            xs = xs.contiguous().view(-1, *xs.shape[-3:])
            y = torch.cat(qt, 0).to(dev)
            logits = model(xq, xs, a.way, a.shot)
            loss = ce(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss at epoch %d iteration %d' % (ep, i))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
            optim.step()
            tot += loss.item() * y.numel(); cor += (logits.argmax(1) == y).sum().item(); n += y.numel()
            if i % 500 == 0:
                print('ep %d it %d loss %.3f acc %.2f tau %.2f' % (ep, i, tot / n, 100 * cor / n,
                                                                   model.tau.item()), flush=True)
        sched.step()
        v, h = validate(model, a, val_eps, dev)
        rec = {'epoch': ep, 'train_loss': tot / n, 'train_acc': 100 * cor / n, 'val': v, 'val_ci': h,
               'tau': model.tau.item(), 'lr': optim.param_groups[0]['lr'], 'min': (time.time() - t0) / 60}
        hist.append(rec)
        msg = json.dumps(rec)
        print(msg, flush=True); print(msg, file=log, flush=True)
        state = {'model': model.state_dict(), 'encoder_model': a.encoder_model, 'routeA_cfg': cfg,
                 'epoch_index': ep, 'best_prec1': max(best, v), 'data_name': a.data_name}
        rng = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
               'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(dict(state, optim=optim.state_dict(), sched=sched.state_dict(), rng=rng),
                   os.path.join(a.outf, 'model_last.pth.tar'))
        if v > best:
            best = v
            torch.save(state, os.path.join(a.outf, 'model_best.pth.tar'))
        json.dump({'best_val': best, 'history': hist}, open(os.path.join(a.outf, 'summary.json'), 'w'),
                  indent=2)
    print('best val %.2f' % best)


if __name__ == '__main__':
    main()
