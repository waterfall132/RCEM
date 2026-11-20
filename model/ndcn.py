import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _robust_svd(Zf):
    if not torch.isfinite(Zf).all():
        raise FloatingPointError('NDM input contains NaN/Inf: training diverged before the SVD')
    try:
        return torch.linalg.svd(Zf, full_matrices=False)
    except RuntimeError:
        pass
    if Zf.is_cuda:
        try:
            return torch.linalg.svd(Zf, full_matrices=False, driver='gesvd')
        except RuntimeError:
            pass
    U, S, Vh = torch.linalg.svd(Zf.double().cpu(), full_matrices=False)
    return U.to(Zf), S.to(Zf), Vh.to(Zf)


def _topr_svd(Zf, r):
    if not torch.isfinite(Zf).all():
        raise FloatingPointError('NDM input contains NaN/Inf: training diverged before the SVD')
    B, L, C = Zf.shape
    Zd = Zf.double()
    if L >= C:
        G = torch.bmm(Zd.transpose(1, 2), Zd)
        lam, V = torch.linalg.eigh(G)
        lam, V = lam.flip(-1)[:, :r], V.flip(-1)[:, :, :r]
        S = lam.clamp_min(0).sqrt()
        V = V.to(Zf.device)
        U = torch.bmm(Zd, V) / S.to(Zf.device).clamp_min(1e-12).unsqueeze(1)
        Vh = V.transpose(1, 2)
    else:
        G = torch.bmm(Zd, Zd.transpose(1, 2))
        lam, U = torch.linalg.eigh(G)
        lam, U = lam.flip(-1)[:, :r], U.flip(-1)[:, :, :r]
        S = lam.clamp_min(0).sqrt()
        U = U.to(Zf.device)
        Vh = torch.bmm(U.transpose(1, 2), Zd) / S.to(Zf.device).clamp_min(1e-12).unsqueeze(-1)
    return U.to(Zf.dtype), S.to(Zf.device, Zf.dtype), Vh.to(Zf.dtype)


def _inv_softplus(y, eps=1e-12):
    return math.log(math.expm1(max(y, eps)))


def make_nonneg(Z, mode='relu', eps=1e-8):
    if mode == 'none':
        return Z
    elif mode == 'relu':
        return F.relu(Z)
    elif mode == 'shift':
        zmin = Z.amin(dim=(1, 2), keepdim=True)
        return Z - torch.clamp(zmin, max=0.0)
    elif mode == 'abs':
        return Z.abs()
    else:
        raise ValueError('unknown nonneg_mode: %s' % mode)


class NDM(nn.Module):

    def __init__(self, rank=5, n_iter=1, eps=1e-8, grad_last_only=True):
        super(NDM, self).__init__()
        self.r = rank
        self.T = n_iter
        self.eps = eps
        self.grad_last_only = grad_last_only

    @torch.no_grad()
    def _nndsvd_init(self, Z):
        B, L, C = Z.shape
        r, eps = self.r, self.eps

        if os.environ.get('NDM_FAST_SVD', '0') == '1':
            U, S, Vh = _topr_svd(Z.float(), r)
        else:
            U, S, Vh = _robust_svd(Z.float())
        U, S, Vh = U[:, :, :r], S[:, :r], Vh[:, :r, :]

        W = torch.zeros(B, L, r, device=Z.device, dtype=Z.dtype)
        H = torch.zeros(B, r, C, device=Z.device, dtype=Z.dtype)

        s0 = torch.sqrt(S[:, 0].clamp_min(eps))
        W[:, :, 0] = s0.unsqueeze(1) * U[:, :, 0].abs()
        H[:, 0, :] = s0.unsqueeze(1) * Vh[:, 0, :].abs()

        for j in range(1, r):
            x, y = U[:, :, j], Vh[:, j, :]
            xp, xn = x.clamp_min(0.0), (-x).clamp_min(0.0)
            yp, yn = y.clamp_min(0.0), (-y).clamp_min(0.0)

            xpn = xp.norm(dim=1) + eps
            xnn = xn.norm(dim=1) + eps
            ypn = yp.norm(dim=1) + eps
            ynn = yn.norm(dim=1) + eps

            mp, mn = xpn * ypn, xnn * ynn
            use_p = (mp >= mn).float().unsqueeze(1)

            u = use_p * (xp / xpn.unsqueeze(1)) + (1 - use_p) * (xn / xnn.unsqueeze(1))
            v = use_p * (yp / ypn.unsqueeze(1)) + (1 - use_p) * (yn / ynn.unsqueeze(1))
            sigma = torch.where(mp >= mn, mp, mn)

            lbd = torch.sqrt((S[:, j] * sigma).clamp_min(eps)).unsqueeze(1)
            W[:, :, j] = lbd * u
            H[:, j, :] = lbd * v

        return W.clamp_min(eps), H.clamp_min(eps)

    def _mu_step(self, Z, W, H):
        eps = self.eps
        num_h = torch.bmm(W.transpose(1, 2), Z)
        den_h = torch.bmm(torch.bmm(W.transpose(1, 2), W), H) + eps
        H = H * (num_h / den_h)

        num_w = torch.bmm(Z, H.transpose(1, 2))
        den_w = torch.bmm(W, torch.bmm(H, H.transpose(1, 2))) + eps
        W = W * (num_w / den_w)
        return W, H

    def forward(self, Z, return_recon=False):
        eps = self.eps
        W, H = self._nndsvd_init(Z)

        if self.grad_last_only:
            with torch.no_grad():
                for _ in range(max(self.T - 1, 0)):
                    W, H = self._mu_step(Z.detach(), W, H)
            W, H = W.detach(), H.detach()
            if self.T >= 1:
                W, H = self._mu_step(Z, W, H)
        else:
            for _ in range(self.T):
                W, H = self._mu_step(Z, W, H)

        A = W / (W.sum(dim=2, keepdim=True) + eps)

        rec_err = None
        if return_recon:
            with torch.no_grad():
                rec = torch.bmm(W, H)
                rec_err = (Z - rec).flatten(1).norm(dim=1) / (Z.flatten(1).norm(dim=1) + eps)
        return A, H, rec_err

    @torch.no_grad()
    def recon_trajectory(self, Z):
        eps = self.eps
        W, H = self._nndsvd_init(Z)
        zn = Z.flatten(1).norm(dim=1) + eps
        traj = [((Z - torch.bmm(W, H)).flatten(1).norm(dim=1) / zn).mean().item()]
        for _ in range(self.T):
            W, H = self._mu_step(Z, W, H)
            traj.append(((Z - torch.bmm(W, H)).flatten(1).norm(dim=1) / zn).mean().item())
        return traj


class MuSigmaFrontend(nn.Module):

    def __init__(self, lam=1.0, temp=10.0):
        super(MuSigmaFrontend, self).__init__()
        self.lam, self.temp = lam, temp

    def forward(self, Z, ref):
        s = F.cosine_similarity(F.normalize(Z, dim=2), F.normalize(ref, dim=1).unsqueeze(1), dim=2)
        mu = s.mean(dim=1, keepdim=True)
        sd = s.std(dim=1, keepdim=True)
        return torch.sigmoid(self.temp * (s - (mu - self.lam * sd)))


class NDCN_Metric(nn.Module):

    def __init__(self, way_num=5, shot_num=1, neighbor_k=3,
                 rank=5, n_iter=1, beta=10.0, margin=0.3,
                 lambda_bg=0.05, lambda_div=0.01, lambda_mask=0.001,
                 score_mode='ref', frontend='ndm', nonneg_mode='relu',
                 alpha_norm='minmax', alpha_tau=8.0, div_margin=0.0,
                 pool_floor=0.05, mask_budget=0.0, uniform_mask=False,
                 use_bg_sep=True, use_qbg=True,
                 use_triplet=True, use_bgloss=True,
                 use_div=True, use_mask=True,
                 grad_last_only=True, mask_mode='ndm', eps=1e-8):
        super(NDCN_Metric, self).__init__()
        if mask_mode not in ('ndm', 'pointwise'):
            raise ValueError('unknown mask_mode: %s' % mask_mode)
        self.mask_mode = mask_mode
        self.way_num, self.shot_num = way_num, shot_num
        self.eps, self.margin = eps, margin
        self.lambda_bg, self.lambda_div, self.lambda_mask = lambda_bg, lambda_div, lambda_mask
        self.score_mode, self.frontend, self.nonneg_mode = score_mode, frontend, nonneg_mode
        self.alpha_norm, self.alpha_tau, self.div_margin = alpha_norm, alpha_tau, div_margin
        self.pool_floor = pool_floor
        self._alpha_range = None
        self.mask_budget = mask_budget
        self.uniform_mask = uniform_mask
        self.use_bg_sep, self.use_qbg = use_bg_sep, use_qbg
        self.use_triplet, self.use_bgloss = use_triplet, use_bgloss
        self.use_div, self.use_mask = use_div, use_mask

        if frontend == 'ndm':
            self.ndm = NDM(rank=rank, n_iter=n_iter, eps=eps, grad_last_only=grad_last_only)
        elif frontend == 'mustd':
            self.ndm = MuSigmaFrontend()
        else:
            raise ValueError('unknown frontend: %s' % frontend)

        self._raw_gamma_fg = nn.Parameter(torch.tensor(_inv_softplus(float(beta))))
        self._raw_gamma_bg = nn.Parameter(torch.tensor(_inv_softplus(float(beta) / 2.0)))
        self.beta_bg = nn.Parameter(torch.tensor(0.5))

        self._diag = {}

    @property
    def gamma_fg(self):
        return F.softplus(self._raw_gamma_fg)

    @property
    def gamma_bg(self):
        return F.softplus(self._raw_gamma_bg)

    @staticmethod
    def _flatten(x):
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1).contiguous()

    def _rescale_alpha(self, a):
        if self.alpha_norm == 'none':
            return a
        if self.alpha_norm == 'minmax':
            lo = a.amin(dim=1, keepdim=True)
            hi = a.amax(dim=1, keepdim=True)
            rng = hi - lo
            flat = rng < 1e-6
            return torch.where(flat, torch.full_like(a, 0.5), (a - lo) / (rng + self.eps))
        if self.alpha_norm == 'zsig':
            return torch.sigmoid(self.alpha_tau * (a - a.mean(dim=1, keepdim=True)))
        if self.alpha_norm in ('setminmax', 'supminmax'):
            if self.alpha_norm == 'supminmax' and self._alpha_range is not None:
                lo, hi = self._alpha_range
            else:
                lo, hi = a.min(), a.max()
            rng = hi - lo
            if float(rng.detach()) < 1e-6:
                return torch.full_like(a, 0.5)
            return ((a - lo) / (rng + self.eps)).clamp(0.0, 1.0)
        raise ValueError('unknown alpha_norm: %s' % self.alpha_norm)

    def _alpha(self, H, ref, A=None, Z=None):
        if self.score_mode == 'ref':
            cos = F.cosine_similarity(F.normalize(H, dim=2),
                                      F.normalize(ref, dim=1).unsqueeze(1), dim=2)
            self._raw_s = 0.5 * (cos + 1.0)
            return self._rescale_alpha(self._raw_s)
        elif self.score_mode == 'centered':
            Hc = H - H.mean(dim=1, keepdim=True)
            rc = ref - ref.mean(dim=0, keepdim=True)
            cos = F.cosine_similarity(F.normalize(Hc, dim=2),
                                      F.normalize(rc, dim=1).unsqueeze(1), dim=2)
            return self._rescale_alpha(0.5 * (cos + 1.0))
        elif self.score_mode == 'free':
            nrm = Z.norm(dim=2)
            num = (A * nrm.unsqueeze(2)).sum(dim=1)
            den = A.sum(dim=1) + self.eps
            raw = num / den
            self._raw_s = raw
            return self._rescale_alpha(raw)
        else:
            raise ValueError('unknown score_mode: %s' % self.score_mode)

    def _mask(self, Z, ref):
        if self.uniform_mask:
            return Z.new_ones(Z.size(0), Z.size(1)), None, None
        if getattr(self, 'mask_mode', 'ndm') == 'pointwise':
            self._raw_s = Z.norm(dim=2)
            M = self._rescale_alpha(self._raw_s)
            return M, None, None
        if self.frontend == 'mustd':
            M = self.ndm(Z, ref)
            return M, None, None
        A, H, _ = self.ndm(Z)
        alpha = self._alpha(H, ref, A=A, Z=Z)
        M = torch.clamp((A * alpha.unsqueeze(1)).sum(dim=2), 0.0, 1.0)
        return M, alpha, A

    @staticmethod
    def _pool(Z, M, eps, floor=0.05):
        L = M.size(1)
        den_fg = M.sum(1, keepdim=True).clamp_min(floor * L)
        den_bg = (1 - M).sum(1, keepdim=True).clamp_min(floor * L)
        fg = (M.unsqueeze(2) * Z).sum(1) / den_fg
        bg = ((1 - M).unsqueeze(2) * Z).sum(1) / den_bg
        return fg, bg

    @staticmethod
    def _cos_dist(a, b):
        return 1.0 - torch.mm(F.normalize(a, dim=1), F.normalize(b, dim=1).t())

    def forward(self, x_query, x_support, target=None):
        eps, way, shot = self.eps, self.way_num, self.shot_num

        Zq = make_nonneg(self._flatten(x_query), self.nonneg_mode, eps)
        Zs = make_nonneg(self._flatten(x_support), self.nonneg_mode, eps)
        NQ, L, C = Zq.shape

        ref_q = Zq.mean(dim=1)
        Zs_cls = Zs.view(way, shot, L, C)
        ref_s_cls = Zs_cls.mean(dim=(1, 2))
        ref_s = ref_s_cls.unsqueeze(1).expand(way, shot, C).reshape(way * shot, C)

        self._alpha_range = None
        Ms, alpha_s, _ = self._mask(Zs, ref_s)
        if self.alpha_norm == 'supminmax' and getattr(self, '_raw_s', None) is not None:
            self._alpha_range = (self._raw_s.min().detach(), self._raw_s.max().detach())
        Mq, alpha_q, _ = self._mask(Zq, ref_q)
        self._alpha_range = None

        q_fg, q_bg = self._pool(Zq, Mq, eps, self.pool_floor)

        Ms_c = Ms.view(way, shot * L)
        Zs_c = Zs.view(way, shot * L, C)
        _M = Ms_c.size(1)
        p_fg = ((Ms_c.unsqueeze(2) * Zs_c).sum(1)
                / Ms_c.sum(1, keepdim=True).clamp_min(self.pool_floor * _M))
        p_bg = (((1 - Ms_c).unsqueeze(2) * Zs_c).sum(1)
                / (1 - Ms_c).sum(1, keepdim=True).clamp_min(self.pool_floor * _M))

        d_fg = self._cos_dist(q_fg, p_fg)
        logit = -self.gamma_fg * d_fg
        d_sep = d_qbg = None
        if self.use_bg_sep:
            d_sep = self._cos_dist(q_fg, p_bg)
            logit = logit + self.gamma_bg * d_sep
        if self.use_qbg:
            d_qbg = self._cos_dist(q_bg, p_bg)
            logit = logit - self.beta_bg * d_qbg

        if getattr(self, 'record_diagnostics', True):
            self._last_Ms = Ms.detach()
            self._last_Mq, self._last_alpha_q = Mq.detach(), (
                alpha_q.detach() if alpha_q is not None else None)

            self._diag = {
                'logit_spread': (logit.max(1).values - logit.min(1).values).mean().detach(),
                'mask_mean_q': Mq.mean().detach(),
                'mask_mean_s': Ms.mean().detach(),
                'alpha_hi': (alpha_q > 0.7).float().sum(1).mean().detach()
                            if alpha_q is not None else torch.tensor(0.),
                'proto_offdiag': self._offdiag_cos(p_fg).detach(),
            }

        if (not self.training) or target is None:
            return logit

        aux = self._aux_losses(q_fg, q_bg, p_fg, p_bg, Mq, target)
        return logit, aux

    @staticmethod
    def _offdiag_cos(P):
        Pn = F.normalize(P, dim=1)
        G = torch.mm(Pn, Pn.t())
        n = G.size(0)
        return (G.sum() - G.diag().sum()) / (n * (n - 1) + 1e-8)

    def _aux_losses(self, q_fg, q_bg, p_fg, p_bg, Mq, target):
        eps, D = self.eps, self.margin
        NQ = q_fg.size(0)
        idx = torch.arange(NQ, device=q_fg.device)

        d_qp_fg = self._cos_dist(q_fg, p_fg)
        d_pos = d_qp_fg[idx, target]

        out = {}

        if self.use_triplet:
            masked = d_qp_fg.clone()
            masked[idx, target] = float('inf')
            d_neg = masked.min(dim=1).values
            out['triplet'] = F.relu(D + d_pos - d_neg).mean()
        else:
            out['triplet'] = q_fg.new_zeros(())

        if self.use_bgloss:
            d_qfg_pbg = self._cos_dist(q_fg, p_bg)[idx, target]
            d_qbg_pbg = self._cos_dist(q_bg, p_bg)[idx, target]
            out['bg'] = (F.relu(D + d_pos - d_qfg_pbg) + 0.5 * d_qbg_pbg).mean()
        else:
            out['bg'] = q_fg.new_zeros(())

        if self.use_div:
            Pn = F.normalize(p_fg, dim=1)
            G = torch.mm(Pn, Pn.t())
            n = G.size(0)
            off = ~torch.eye(n, dtype=torch.bool, device=G.device)
            if self.div_margin > 0:
                out['div'] = (F.relu(G[off] - self.div_margin) ** 2).mean()
            else:
                I = torch.eye(n, device=G.device, dtype=G.dtype)
                out['div'] = ((G - I) ** 2).mean()
        else:
            out['div'] = q_fg.new_zeros(())

        if not self.use_mask or self.uniform_mask:
            out['mask'] = q_fg.new_zeros(())
        elif self.mask_budget > 0:
            out['mask'] = F.relu(Mq.mean() - self.mask_budget)
        else:
            out['mask'] = Mq.mean()

        out['total'] = (out['triplet']
                        + self.lambda_bg * out['bg']
                        + self.lambda_div * out['div']
                        + self.lambda_mask * out['mask'])
        return out
