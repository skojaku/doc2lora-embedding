"""Per-cell numbers behind Fig. 2(c), the two-corner mixing colorband.

For one A-B pair decoded at the 13 grid points along the edge (weights on B =
0, 1/12, ..., 1), compute for every method and cell:

  mix_t      SBERT projection of the decoded abstract onto the A->B axis,
             dot(d - A, B - A) / |B - A|^2 (same statistic as
             workflow/scripts/pair_axis_metrics.py; not clipped)
  copy_rate  fraction of the abstract's word tokens that occur verbatim in
             either source lead (same tokeniser as pair_axis_copyrate.py)
  kw         the decoded keyword string

plus the two corners' titles and DOIs (from the corner spec; else looked up via
the pair manifest + APS text table when both are available), so the figure
script needs neither the GPU decodes nor the corpus. SBERT (all-mpnet-base-v2) runs on CPU in seconds.

Dual mode: driven by Snakemake (workflow/rules/fig2_pacs_clustering.smk) or

    python workflow/scripts/fig2_colorband_metrics.py \
        --corners data/pair_axis/corners_pairCSML_00.json \
        --decodes data/pair_axis/results/absfollow_pairCSML_00_pairaxis.json \
        --icae    data/pair_axis/results/absfollow_pairCSML_00_pairaxis_icae.json \
        --out     data/pair_axis/results/colorband_pairCSML_00_pairaxis.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

METHODS = ["doc2lora", "icae", "incontext"]


def toks(t):
    return re.findall(r"[a-z][a-z-]{2,}", (t or "").lower())


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def load_cells(base_path, icae_path):
    """Merge the doc2lora/in-context decode file with the ICAE one, keyed by bary."""
    cells = {}
    for path, keep in [(base_path, ("doc2lora", "incontext")), (icae_path, ("icae",))]:
        if not path or not os.path.exists(path):
            continue
        for c in json.load(open(path))["cells"]:
            rec = cells.setdefault(tuple(c["bary"]), {})
            rec.update({k: v for k, v in c.items() if k in keep})
    return cells


def corner_meta(spec, set_name, manifest_path, paper_text_path):
    """Title + DOI per corner. The corner spec may carry `paper_id` / `doi`
    itself (corners_pairCSML_00.json does); otherwise the paper id comes from
    the pair manifest and the DOI from the APS text table, when those exist on
    this machine, and the spec's name is used as the title."""
    meta = {k: {"name": spec["corners"][k]["name"],
                "doi": spec["corners"][k].get("doi"),
                "paper_id": spec["corners"][k].get("paper_id")}
            for k in spec["keys"]}
    if all(meta[k]["doi"] for k in spec["keys"]):
        return meta
    if not (manifest_path and os.path.exists(manifest_path)):
        return meta
    rec = next((m for m in json.load(open(manifest_path)) if m["set"] == set_name), None)
    if rec is None:
        return meta
    for k, pid in zip(spec["keys"], rec["paper_ids"]):
        meta[k]["paper_id"] = int(pid)
    if not (paper_text_path and os.path.exists(paper_text_path)):
        return meta
    import pandas as pd
    df = pd.read_parquet(paper_text_path, columns=["aps_paper_id", "doi", "title"])
    df = df.set_index("aps_paper_id")
    for k in spec["keys"]:
        pid = meta[k]["paper_id"]
        if pid in df.index:
            meta[k]["doi"] = str(df.loc[pid, "doi"])
            meta[k]["name"] = str(df.loc[pid, "title"])
    return meta


def main(corners, decodes, icae, out, manifest=None, paper_text=None, device="cpu"):
    spec = json.load(open(corners))
    keys = spec["keys"]
    kA, kB = keys[0], keys[1]
    leadA, leadB = spec["corners"][kA]["lead"], spec["corners"][kB]["lead"]
    set_name = spec.get("set") or os.path.basename(corners)

    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("all-mpnet-base-v2", device=device)

    def emb(t):
        return unit(np.atleast_2d(
            sb.encode(t, normalize_embeddings=True, show_progress_bar=False)).ravel())

    eA, eB = emb(leadA), emb(leadB)
    axis = eB - eA
    denom = float(axis @ axis) + 1e-12
    srctok = set(toks(leadA)) | set(toks(leadB))

    cells = load_cells(decodes, icae)
    R = max(sum(b) for b in cells)          # grid denominator (12)
    methods = {}
    for m in METHODS:
        rows = []
        for k in range(R + 1):
            rec = cells.get((R - k, k, 0), {}).get(m)
            if rec is None:
                continue
            ab, kw = rec.get("abs", ""), rec.get("kw", "")
            at = toks(ab)
            rows.append({
                "k": k,
                "weight_b": k / R,
                "mix_t": float((emb(ab) - eA) @ axis / denom) if ab.strip() else None,
                "copy_rate": (sum(w in srctok for w in at) / len(at)) if at else None,
                "kw": kw,
            })
        methods[m] = rows

    result = {
        "set": set_name,
        "grid_den": R,
        "sbert": "all-mpnet-base-v2",
        "corners": corner_meta(spec, set_name, manifest, paper_text),
        "methods": methods,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(result, open(out, "w"), indent=1, ensure_ascii=False)
    print("wrote", out)


if "snakemake" in sys.modules:
    main(corners=snakemake.input["corners"],
         decodes=snakemake.input["decodes"],
         icae=snakemake.input["icae"],
         out=snakemake.output["metrics"],
         manifest=snakemake.params["manifest"],
         paper_text=snakemake.params["paper_text"],
         device=snakemake.params["device"])
elif __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", required=True)
    ap.add_argument("--decodes", required=True, help="absfollow_<set>_pairaxis.json (doc2lora + in-context)")
    ap.add_argument("--icae", default=None, help="absfollow_<set>_pairaxis_icae.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="pair_manifest.json (paper ids)")
    ap.add_argument("--paper-text", default=None, help="data/aps/paper_text.parquet (titles + DOIs)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    main(a.corners, a.decodes, a.icae, a.out, a.manifest, a.paper_text, a.device)
