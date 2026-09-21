"""Shared helpers for the S2AND author-name-disambiguation test with idea genes."""
import itertools
import numpy as np


def b3(true_clus, pred_clus):
    """B^3 precision/recall/F1 (faithful port of s2and.eval.b3_precision_recall_fscore).
    true_clus/pred_clus: {cluster_id: iterable of signature_ids}. Must cover the same sig set."""
    tcset = set(itertools.chain.from_iterable(true_clus.values()))
    pcset = set(itertools.chain.from_iterable(pred_clus.values()))
    assert tcset == pcset, f"coverage mismatch {len(tcset)} vs {len(pcset)}"
    true = {k: frozenset(v) for k, v in true_clus.items()}
    pred = {k: frozenset(v) for k, v in pred_clus.items()}
    rev_t = {vi: k for k, v in true.items() for vi in v}
    rev_p = {vi: k for k, v in pred.items() for vi in v}
    inter = {}
    P = R = 0.0
    n = len(tcset)
    for item in tcset:
        pc = pred[rev_p[item]]; tc = true[rev_t[item]]
        key = (rev_p[item], rev_t[item])
        if key not in inter:
            inter[key] = len(pc & tc)
        P += inter[key] / len(pc)
        R += inter[key] / len(tc)
    P /= n; R /= n
    F = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)
    return P, R, F


def b3_per_sig(true_clus, pred_clus):
    """Like b3 but also returns per-signature (precision, recall) for bootstrap over signatures.
    Returns (P, R, F, dict{signature_id: (p, r)})."""
    tcset = set(itertools.chain.from_iterable(true_clus.values()))
    pcset = set(itertools.chain.from_iterable(pred_clus.values()))
    assert tcset == pcset
    true = {k: frozenset(v) for k, v in true_clus.items()}
    pred = {k: frozenset(v) for k, v in pred_clus.items()}
    rev_t = {vi: k for k, v in true.items() for vi in v}
    rev_p = {vi: k for k, v in pred.items() for vi in v}
    inter = {}; per = {}; P = R = 0.0; n = len(tcset)
    for item in tcset:
        pc = pred[rev_p[item]]; tc = true[rev_t[item]]
        key = (rev_p[item], rev_t[item])
        if key not in inter:
            inter[key] = len(pc & tc)
        p = inter[key] / len(pc); r = inter[key] / len(tc)
        per[item] = (p, r); P += p; R += r
    P /= n; R /= n
    F = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)
    return P, R, F, per
