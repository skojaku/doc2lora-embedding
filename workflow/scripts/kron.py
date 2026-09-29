"""Kronecker-factored invertible task adapter on top of frozen doc2lora idea genes.

Gene per paper = tensor G in R^{L x 512} (L=26 gemma layers, mean-pooled over rank/query).
Transform:  Z[l,:] = a_l * (G[l,:] @ C),   C = Q diag(exp(s)),  a_l = exp(alpha_l)
  - A = diag(exp(alpha))  : per-LAYER scaling (L params)            [diagonal, as decided]
  - B = I                 : query axis already pooled away in the npz
  - C = Q Lambda          : 512x512 rotation (orthogonal, matrix_exp of skew) x positive diagonal
Exactly invertible factor-wise; near-identity at init.  Trained with SPECTER2-style citation
contrastive loss (in-batch InfoNCE) + a simple log-det volume regularizer (kept simple, as decided).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KronAdapter(nn.Module):
    """Invertible factor map on frozen genes G in R^{L x d}.

    Shared (default, ~d^2 params): ONE hidden transform C = Q diag(exp(s)) applied to every
    layer, differentiated only by a per-layer scalar a_l = exp(alpha_l).
        Z[l,:] = a_l * (G[l,:] @ C)

    per_layer=True (~L*d^2 params): each layer gets its OWN hidden transform
    C_l = Q_l diag(exp(s_l)), so P -> [L,d,d], s -> [L,d]:
        Z[l,:] = a_l * (G[l,:] @ C_l)
    Still exactly invertible factor-wise (Q_l orthogonal, exp(s_l) positive), so the
    decode/round-trip path is preserved (see decodability.py, which dispatches on P.dim()).
    Only matters for multi-layer genes (L>1); for flat text (L=1) it equals the shared map.
    """

    def __init__(self, L=26, d=512, per_layer=False):
        super().__init__()
        self.L, self.d, self.per_layer = L, d, per_layer
        if per_layer:
            self.P = nn.Parameter(torch.zeros(L, d, d))   # per-layer skew -> Q_l = expm(P_l-P_l^T)
            self.s = nn.Parameter(torch.zeros(L, d))      # per-layer hidden log-scales -> Lambda_l
        else:
            self.P = nn.Parameter(torch.zeros(d, d))      # skew generator -> Q = expm(P-P^T)  (Q=I at init)
            self.s = nn.Parameter(torch.zeros(d))         # hidden log-scales  -> Lambda = exp(s)  (I at init)
        self.alpha = nn.Parameter(torch.zeros(L))         # per-layer log-scales -> A = exp(alpha)  (I at init)

    def C(self):
        if self.per_layer:
            W = self.P - self.P.transpose(-1, -2)         # [L,d,d] skew-symmetric
            Q = torch.matrix_exp(W)                       # [L,d,d] orthogonal, det=+1
            return Q * torch.exp(self.s)[:, None, :]      # C_l = Q_l @ diag(exp(s_l))
        W = self.P - self.P.t()                           # skew-symmetric
        Q = torch.matrix_exp(W)                           # orthogonal, det=+1
        return Q * torch.exp(self.s)[None, :]             # C = Q @ diag(exp(s))

    def forward(self, G):                                 # G: [B, L, d]
        eq = "bld,lde->ble" if self.per_layer else "bld,de->ble"
        Z = torch.einsum(eq, G, self.C())                 # rotate+scale hidden axis
        Z = Z * torch.exp(self.alpha)[None, :, None]      # per-layer scale
        return Z.reshape(G.shape[0], -1)                  # [B, L*d]

    def logdet_reg(self):
        # simple volume penalty: keep net log-volume of each non-orthogonal factor near 0
        # log|det Lambda| = sum(s) ; log|det A| = sum(alpha)
        return self.s.sum() ** 2 + self.alpha.sum() ** 2


def info_nce(za, zp, tau=0.05):
    za = F.normalize(za, dim=1); zp = F.normalize(zp, dim=1)
    logits = za @ zp.t() / tau                        # in-batch negatives
    labels = torch.arange(za.shape[0], device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
