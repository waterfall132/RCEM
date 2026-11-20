import math

import torch
import torch.nn.functional as F

ASSIGNS = ('ndm', 'point', 'smooth')
SCORES = ('disc_lse',
          'disc_lin',
          'same',
          'norm',
          'random',
          'uniform')
CLASS_INDEPENDENT = ('norm', 'random', 'uniform')


def flatten(x):
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).transpose(1, 2).contiguous()


def image_components(Z, assign='ndm', ndm=None, hw=None, eps=1e-8):
    B, L, C = Z.shape
    g = Z.mean(1)
    out = {'g': g, 'L': L}
    if assign == 'ndm':
        if ndm is None:
            raise ValueError('assign=ndm needs an NDM instance')
        A, _, _ = ndm(Z)
        m = A.sum(1)
        ez = torch.bmm(A.transpose(1, 2), Z)
        e = ez / (m.unsqueeze(-1) + eps)
        s = (A * Z.norm(dim=2, keepdim=True)).sum(1) / (m + eps)
        out.update(e=e, m=m, s=s, ez=ez)
    elif assign == 'point':
        out.update(e=Z, m=Z.new_ones(B, L), s=Z.norm(dim=2))
    elif assign == 'smooth':
        if hw is None:
            raise ValueError('assign=smooth needs the feature-map size hw=(H,W)')
        H, W = hw
        x = Z.transpose(1, 2).reshape(B, C, H, W)
        xs = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        ns = F.avg_pool2d(Z.norm(dim=2).reshape(B, 1, H, W), 3, 1, 1,
                          count_include_pad=False)
        out.update(e=xs.flatten(2).transpose(1, 2), m=Z.new_ones(B, L),
                   s=ns.flatten(1))
    else:
        raise ValueError('unknown assign: %s' % assign)
    return out


def gap_prototypes(g_s, way, shot):
    u = F.normalize(g_s, dim=-1).reshape(way, shot, -1).mean(1)
    return F.normalize(u, dim=-1)


def class_scores(ehat, P, score, t=0.1):
    if P.dim() == 2:
        S = torch.einsum('brc,nc->brn', ehat, P)
    else:
        S = torch.einsum('brc,bnc->brn', ehat, P)
    N = S.shape[-1]
    if score == 'same':
        return S
    if N < 2:
        raise ValueError('discriminative scores need at least two classes')
    if score == 'disc_lin':
        return S - (S.sum(-1, keepdim=True) - S) / (N - 1)
    if score == 'disc_lse':
        E = (S / t).unsqueeze(-2).expand(*S.shape, N)
        eye = torch.eye(N, dtype=torch.bool, device=S.device)
        lse = torch.logsumexp(E.masked_fill(eye, float('-inf')), dim=-1)
        return S - t * (lse - math.log(N - 1))
    raise ValueError('not a class-conditional score: %s' % score)


def component_weights(d, valid, T):
    z = d.unsqueeze(0) / T.view(-1, 1, 1, 1)
    z = z.masked_fill(~valid.unsqueeze(0).unsqueeze(-1), float('-inf'))
    return torch.softmax(z, dim=2)


def _scores_for(comp, ehat, valid, P, score, t, support_s_mean=None, gen=None):
    if score in ('disc_lse', 'disc_lin', 'same'):
        return class_scores(ehat, P, score, t), 'cls'
    if score == 'norm':
        ref = support_s_mean if support_s_mean is not None else comp['s'].mean()
        return (comp['s'] / (ref + 1e-8)).unsqueeze(-1), 'ind'
    if score == 'random':
        B, R = valid.shape
        N = P.shape[-2]
        r = torch.randn(B, R, N, generator=gen, device='cpu').to(ehat.device)
        return r, 'cls'
    if score == 'uniform':
        return None, 'uniform'
    raise ValueError('unknown score: %s' % score)


def local_vectors(comp, P, score, T, t=0.1, support_s_mean=None, gen=None,
                  own_class=None, eps=1e-8):
    e = comp['e']
    ehat = F.normalize(e, dim=-1, eps=eps)
    valid = comp['m'] > 1e-6 * comp['m'].amax(dim=1, keepdim=True).clamp_min(eps)
    d, kind = _scores_for(comp, ehat, valid, P, score, t, support_s_mean, gen)
    if kind == 'uniform':
        w = valid.float() / valid.float().sum(1, keepdim=True).clamp_min(1.0)
        w = w.unsqueeze(0).unsqueeze(-1)
    else:
        if own_class is not None and kind == 'cls':
            d = d.gather(2, own_class.view(-1, 1, 1).expand(-1, d.shape[1], 1))
        w = component_weights(d, valid, T)
    vL = torch.einsum('tbrn,brc->tbnc', w, ehat)
    return F.normalize(vL, dim=-1, eps=eps)


def mix_with_global(g, vL, gamma, eps=1e-8):
    gh = F.normalize(g, dim=-1, eps=eps)[None, None, :, None, :]
    gm = gamma.view(-1, 1, 1, 1, 1)
    return F.normalize((1 - gm) * gh + gm * vL.unsqueeze(0), dim=-1, eps=eps)


def loo_prototypes(g_s, way, shot):
    if shot < 2:
        raise ValueError('leave-one-image-out needs shot >= 2')
    u = F.normalize(g_s, dim=-1).reshape(way, shot, -1)
    full = F.normalize(u.mean(1), dim=-1)
    loo = F.normalize((u.sum(1, keepdim=True) - u) / (shot - 1), dim=-1)
    P = full.unsqueeze(0).expand(way * shot, way, -1).clone()
    idx = torch.arange(way * shot, device=g_s.device)
    cls = idx // shot
    P[idx, cls] = loo.reshape(way * shot, -1)
    return P, cls


def route_a_logits(cq, cs, way, shot, score='disc_lse', T=None, gamma=None, t=0.1,
                   proto='gap', gen=None, eps=1e-8):
    dev = cq['g'].device
    T = torch.as_tensor(T if T is not None else [0.1], dtype=cq['g'].dtype, device=dev).flatten()
    gamma = torch.as_tensor(gamma if gamma is not None else [0.5], dtype=cq['g'].dtype,
                            device=dev).flatten()
    P = gap_prototypes(cs['g'], way, shot)
    s_ref = cs['s'][cs['m'] > 0].mean() if score == 'norm' else None
    vq = mix_with_global(cq['g'], local_vectors(cq, P, score, T, t, s_ref, gen),
                         gamma)
    if proto == 'gap':
        return (vq * P[None, None, None]).sum(-1)
    if proto == 'loo':
        Ps, cls = loo_prototypes(cs['g'], way, shot)
        vs = local_vectors(cs, Ps, score, T, t, s_ref, gen, own_class=cls)
        vs = mix_with_global(cs['g'], vs, gamma)[..., 0, :]
        nG, nT = vs.shape[:2]
        Pn = F.normalize(vs.reshape(nG, nT, way, shot, -1).mean(3), dim=-1, eps=eps)
        return (vq * Pn.unsqueeze(2)).sum(-1)
    raise ValueError('unknown proto mode: %s' % proto)


def gap_logits(cq, cs, way, shot):
    P = gap_prototypes(cs['g'], way, shot)
    return F.normalize(cq['g'], dim=-1) @ P.T


def ndcp_original_logits(cq, cs, way, shot, rho0=0.05, eps=1e-8):
    ss = cs['s']
    lo, hi = ss.min(), ss.max()
    rng = hi - lo

    def alpha(s):
        if float(rng) < 1e-6:
            return torch.full_like(s, 0.5)
        return ((s - lo) / (rng + eps)).clamp(0.0, 1.0)

    L = cq['L']
    aq = alpha(cq['s'])
    num_q = (aq.unsqueeze(-1) * cq['ez']).sum(1)
    den_q = (aq * cq['m']).sum(1, keepdim=True)
    vq = num_q / den_q.clamp_min(rho0 * L)
    vq = torch.where(den_q > 0, vq, cq['g'])

    a_s = alpha(ss)
    num_s = (a_s.unsqueeze(-1) * cs['ez']).sum(1).reshape(way, shot, -1).sum(1)
    den_s = (a_s * cs['m']).sum(1).reshape(way, shot).sum(1, keepdim=True)
    p = num_s / den_s.clamp_min(rho0 * shot * L)
    p = torch.where(den_s > 0, p, cs['g'].reshape(way, shot, -1).mean(1))
    return F.normalize(vq, dim=-1, eps=eps) @ F.normalize(p, dim=-1, eps=eps).T
