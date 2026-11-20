import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ndcn import NDCN_Metric, make_nonneg


class NDCN_Local_Metric(NDCN_Metric):

    def __init__(self, *args, neighbor_k=3, chunk_size=0, **kwargs):
        super(NDCN_Local_Metric, self).__init__(*args, **kwargs)
        self.neighbor_k = neighbor_k
        self.chunk_size = chunk_size

    @staticmethod
    def _local_knn(cos, w_sup, k, eps):
        ws = w_sup.unsqueeze(0).unsqueeze(2)
        idx = (cos * ws).topk(k, dim=3).indices
        c_sel = cos.gather(3, idx)
        w_sel = ws.expand(cos.shape).gather(3, idx)
        return (c_sel * w_sel).sum(3) / (w_sel.sum(3) + eps)

    @staticmethod
    def _aggregate(local, w_qry, eps):
        num = (local * w_qry.unsqueeze(1)).sum(2)
        den = w_qry.sum(1, keepdim=True) + eps
        return num / den

    def _auto_chunk(self, NQ, Lq, Ms):
        if self.chunk_size > 0:
            return self.chunk_size
        per_query = self.way_num * Lq * Ms * 4
        budget = 256 * 1024 * 1024
        return max(1, min(NQ, budget // max(per_query, 1)))

    def _scores(self, Qn, Sn, Mq, Ms):
        eps, k = self.eps, self.neighbor_k
        NQ, Lq = Qn.size(0), Qn.size(1)
        step = self._auto_chunk(NQ, Lq, Sn.size(1))
        need_bg = self.use_bg_sep or self.use_qbg
        Ms_bg = 1.0 - Ms

        S_fg, S_sep, S_qbg = [], [], []
        for i in range(0, NQ, step):
            q = Qn[i:i + step]
            cos = torch.einsum('qlc,wmc->qwlm', q, Sn)
            mq = Mq[i:i + step]

            S_fg.append(self._aggregate(self._local_knn(cos, Ms, k, eps), mq, eps))
            if need_bg:
                local_bg = self._local_knn(cos, Ms_bg, k, eps)
                if self.use_bg_sep:
                    S_sep.append(self._aggregate(local_bg, mq, eps))
                if self.use_qbg:
                    S_qbg.append(self._aggregate(local_bg, 1.0 - mq, eps))
                del local_bg
            del cos

        cat = lambda x: torch.cat(x, 0) if x else None
        return cat(S_fg), cat(S_sep), cat(S_qbg)

    def forward(self, x_query, x_support, target=None):
        eps, way, shot = self.eps, self.way_num, self.shot_num

        Zq = make_nonneg(self._flatten(x_query), self.nonneg_mode, eps)
        Zs = make_nonneg(self._flatten(x_support), self.nonneg_mode, eps)
        NQ, Lq, C = Zq.shape
        Ls = Zs.size(1)

        ref_q = Zq.mean(dim=1)
        Zs_cls = Zs.view(way, shot, Ls, C)
        ref_s = Zs_cls.mean(dim=(1, 2)).unsqueeze(1).expand(way, shot, C).reshape(way * shot, C)

        Mq, alpha_q, _ = self._mask(Zq, ref_q)
        Ms, alpha_s, _ = self._mask(Zs, ref_s)

        Qn = F.normalize(Zq, dim=2)
        Sn = F.normalize(Zs, dim=2).view(way, shot * Ls, C)
        Ms_c = Ms.view(way, shot * Ls)

        S_fg, S_sep, S_qbg = self._scores(Qn, Sn, Mq, Ms_c)
        self._cache = (Qn, Sn, Mq, Ms_c)

        logit = self.gamma_fg * S_fg
        if self.use_bg_sep:
            logit = logit - self.gamma_bg * S_sep
        if self.use_qbg:
            logit = logit + self.beta_bg * S_qbg

        Zs_c = Zs.view(way, shot * Ls, C)
        p_fg = (Ms_c.unsqueeze(2) * Zs_c).sum(1) / (Ms_c.sum(1, keepdim=True) + eps)

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

        d_fg = 1.0 - S_fg
        d_sep = (1.0 - S_sep) if self.use_bg_sep else None
        d_qbg = (1.0 - S_qbg) if self.use_qbg else None
        aux = self._aux_from_dists(d_fg, d_sep, d_qbg, Mq, p_fg, target)
        return logit, aux

    def _loss_scores(self, Qn, Sn, Mq, Ms):
        eps, k = self.eps, self.neighbor_k
        NQ, Lq = Qn.size(0), Qn.size(1)
        step = self._auto_chunk(NQ, Lq, Sn.size(1))
        Ms_bg = 1.0 - Ms
        sep, qbg = [], []
        for i in range(0, NQ, step):
            cos = torch.einsum('qlc,wmc->qwlm', Qn[i:i + step], Sn)
            local_bg = self._local_knn(cos, Ms_bg, k, eps)
            mq = Mq[i:i + step]
            sep.append(self._aggregate(local_bg, mq, eps))
            qbg.append(self._aggregate(local_bg, 1.0 - mq, eps))
            del cos, local_bg
        return torch.cat(sep, 0), torch.cat(qbg, 0)

    def _aux_from_dists(self, d_fg, d_sep, d_qbg, Mq, p_fg, target):
        D = self.margin
        NQ = d_fg.size(0)
        idx = torch.arange(NQ, device=d_fg.device)
        d_pos = d_fg[idx, target]
        out = {}

        if self.use_triplet:
            masked = d_fg.clone()
            masked[idx, target] = float('inf')
            d_neg = masked.min(dim=1).values
            out['triplet'] = F.relu(D + d_pos - d_neg).mean()
        else:
            out['triplet'] = d_fg.new_zeros(())

        if self.use_bgloss:
            S_sep_l, S_qbg_l = self._loss_scores(*self._cache)
            t1 = F.relu(D + d_pos - (1.0 - S_sep_l)[idx, target])
            t2 = 0.5 * (1.0 - S_qbg_l)[idx, target]
            out['bg'] = (t1 + t2).mean()
        else:
            out['bg'] = d_fg.new_zeros(())

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
            out['div'] = d_fg.new_zeros(())

        out['mask'] = Mq.mean() if self.use_mask else d_fg.new_zeros(())

        out['total'] = (out['triplet']
                        + self.lambda_bg * out['bg']
                        + self.lambda_div * out['div']
                        + self.lambda_mask * out['mask'])
        return out
