import argparse
import glob
import json
import math
import os
import re
import sys
import zipfile

import numpy as np

PAT = re.compile(r'_(\d)s_dn4(_[a-z0-9]+)?_(B_eta05|gap)_s(\d+)')


def lit(x):
    return float(np.mean([1.96 * r.std(ddof=1) / math.sqrt(r.size) for r in np.array_split(x, 5)]))


def pvals(d):
    if not d.any():
        return float('nan'), float('nan')
    t = d.mean() / (d.std(ddof=1) / math.sqrt(d.size))
    pt = math.erfc(abs(t) / math.sqrt(2))
    try:
        from scipy.stats import wilcoxon
        pw = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 10 else float('nan')
    except Exception:
        pw = float('nan')
    return pt, pw


def diff(a, b):
    d = np.round(a - b, 4)
    h = 1.96 * d.std(ddof=1) / math.sqrt(d.size)
    pt, pw = pvals(d)
    return '%+.2f [%.2f, %.2f]  paired t p=%.2g  Wilcoxon p=%.2g  (RCEM better in %d / worse in %d episodes)' % (
        d.mean(), d.mean() - h, d.mean() + h, pt, pw, (d > 0).sum(), (d < 0).sum())


def acc(x):
    pooled = 1.96 * x.std(ddof=1) / math.sqrt(x.size)
    return '%.2f +- %.2f (95%% CI, all %d episodes)   [per-round convention +- %.2f]' % (x.mean(), pooled, x.size, lit(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    unz = os.path.join(root, 'unzipped')
    for z in sorted(glob.glob(os.path.join(root, '*.zip'))):
        dst = os.path.join(unz, os.path.splitext(os.path.basename(z))[0])
        if not os.path.isdir(dst):
            print('extracting', z)
            with zipfile.ZipFile(z) as f:
                f.extractall(dst)

    groups = {}
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if not fn.endswith('.npz'):
                continue
            p = os.path.join(dp, fn)
            parts = p.replace('\\', '/').split('/')
            if len(parts) < 3 or not parts[-2].startswith('test_'):
                continue
            ev = parts[-3]
            m = PAT.search(ev)
            if not m:
                continue
            K, tag, model, seed = int(m.group(1)), m.group(2) or '', m.group(3), int(m.group(4))
            setting = parts[-4]
            key = (setting, K, tag, seed)
            g = groups.setdefault(key, {})
            g.setdefault(model, {'dir': os.path.dirname(dp), 'npz': {}})['npz'][fn[:-4]] = p

    lines = []
    for key in sorted(groups):
        setting, K, tag, seed = key
        g = groups[key]
        lines.append('=' * 100)
        lines.append('%s | %d-shot | %s | seed %d' % (setting, K, ('variant ' + tag[1:]) if tag else 'main', seed))
        b, gp = g.get('B_eta05'), g.get('gap')
        if b:
            js = glob.glob(os.path.join(b['dir'], 'results_*shot.json'))
            if js:
                c = json.load(open(js[0], encoding='utf-8'))
                ch = c['choices'][c['selected']]
                lines.append('  selected on validation: tau_G=%.2f tau_L=%.2f lambda_0=%s%s' % (
                    ch['tau_G'], ch['tau_L'], ch['lam_fixed'], ('  zeta=%s' % ch['zeta']) if 'zeta' in ch else ''))
        head = 'B_select' if K >= 2 else 'B_fixed'
        R = np.load(b['npz'][head])['accuracy'].astype(float).ravel() if b and head in b['npz'] else None
        RG = np.load(b['npz']['gap'])['accuracy'].astype(float).ravel() if b and 'gap' in b['npz'] else None
        G = np.load(gp['npz']['gap'])['accuracy'].astype(float).ravel() if gp and 'gap' in gp['npz'] else None
        if R is not None:
            lines.append('  RCEM (fused)            %s   (%d episodes)' % (acc(R), R.size))
        if RG is not None:
            lines.append('  RCEM, global branch     %s' % acc(RG))
        if G is not None:
            lines.append('  GAP + CE                %s' % acc(G))
        if R is not None and G is not None and R.shape == G.shape:
            lines.append('  RCEM - GAP + CE                 %s' % diff(R, G))
            if RG is not None:
                lines.append('  RCEM global branch - GAP + CE   %s' % diff(RG, G))
                lines.append('  RCEM fused - RCEM global branch %s' % diff(R, RG))
        missing = [n for n, v in (('RCEM', b), ('GAP + CE', gp)) if v is None]
        if missing:
            lines.append('  (not found in these folders: %s)' % ', '.join(missing))
    txt = '\n'.join(lines) if lines else 'no result files found below %s' % root
    print(txt)
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(txt + '\n')
        print('\nsaved to', a.out)


if __name__ == '__main__':
    main()
