"""Label each PACS node with vec2text (GTR embedding inversion).

For fairness with the ICAE / KeyLLM baselines, each node's members are retrieved
the same way (SBERT all-mpnet centroid-nearest, TOPK members). Their title+abstract
text is embedded with vec2text's OWN GTR encoder (call_embedding_model -- NOT a
standalone SentenceTransformer, which produces gibberish), mean-pooled into a node
centroid, renormalized, and inverted with the gtr-base corrector.

Output: vec2text_labels.json  {node_code: {"label": ..., "medoid": ...}}

Run (GPU): CUDA_VISIBLE_DEVICES=0 python decode_vec2text_nodes.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]   # repository root, not a fixed ~/projects path
RES = ROOT / "data/pacs/results"
HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes


TOPK = 20          # centroid-nearest members embedded per node
NUM_STEPS = 20     # vec2text correction steps
MAX_LEN = 128      # GTR truncation (matches vec2text gtr-base training)

# field cluster-id -> divisions -> subfields  (identical to the doc2lora figure)
SPEC = [
    {"fid": 3, "divs": [{"c": "75", "subs": ["75.10", "75.30"]},
                        {"c": "71", "subs": ["71.10", "71.20"]}]},
    {"fid": 0, "divs": [{"c": "03", "subs": ["03.67", "03.65"]},
                        {"c": "05", "subs": ["05.45", "05.40"]}]},
    {"fid": 6, "divs": [{"c": "11", "subs": ["11.15", "11.10"]},
                        {"c": "12", "subs": ["12.38", "12.60"]}]},
    {"fid": 2, "divs": [{"c": "42", "subs": ["42.50", "42.65"]},
                        {"c": "47", "subs": ["47.27", "47.20"]}]},
]


def build_node_members():
    """Return ordered list of (code, member_pid_array)."""
    pg = pd.read_parquet(RES / "paper_groups.parquet")
    nodes = []
    for f in SPEC:
        fid = f["fid"]
        nodes.append((str(fid), pg[pg.main_class_id == fid].paper_id.values))
        for d in f["divs"]:
            nodes.append((d["c"], pg[pg.division == d["c"]].paper_id.values))
            for s in d["subs"]:
                nodes.append((s, pg[pg.subdivision == s].paper_id.values))
    return nodes


def load_text():
    txt = pd.read_parquet(
        ROOT / "data/aps/paper_text_pid.parquet",
        columns=["paper_id", "title", "abstract", "text"],
    ).set_index("paper_id")
    return txt


def doc_text(txt, pid):
    try:
        row = txt.loc[pid]
    except KeyError:
        return None
    title = (row["title"] or "").strip()
    abs_txt = (row["abstract"] or "").strip()
    if not abs_txt:
        t = row["text"] or ""
        abs_txt = t.split("Abstract:", 1)[-1].strip() if "Abstract:" in t else ""
    s = (title + ". " + abs_txt).strip(". ").strip()
    return s if s else None


def main():
    import vec2text

    print("loading SBERT embeddings for centroid retrieval...")
    z = np.load(ROOT / "data/aps/embeddings/sbert_allmpnet.npz")
    pids = z["paper_ids"].astype(np.int64)
    vecs = z["vecs"].astype(np.float32)
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    pid2row = {int(p): i for i, p in enumerate(pids)}

    txt = load_text()
    nodes = build_node_members()

    print("loading vec2text gtr-base corrector...")
    corr = vec2text.load_pretrained_corrector("gtr-base")
    model = corr.inversion_trainer.model
    tok = corr.embedder_tokenizer
    dev = model.device

    def gtr_embed(texts):
        embs = []
        for t in texts:
            ins = tok([t], return_tensors="pt", truncation=True,
                      max_length=MAX_LEN, padding=True).to(dev)
            with torch.no_grad():
                e = model.call_embedding_model(
                    input_ids=ins["input_ids"],
                    attention_mask=ins["attention_mask"],
                )
            embs.append(e)
        return torch.cat(embs, dim=0)  # [n, 768]

    out = {}
    for code, member_pids in nodes:
        rows = np.array([pid2row[int(p)] for p in member_pids if int(p) in pid2row])
        if len(rows) < 2:
            out[code] = {"label": "", "medoid": "", "n_used": 0}
            print(f"  {code:>6}: no members")
            continue
        V = vn[rows]
        c = V.mean(0)
        c /= np.linalg.norm(c) + 1e-9
        order = rows[np.argsort(-(V @ c))[:TOPK]]
        texts = [doc_text(txt, int(pids[i])) for i in order]
        texts = [t for t in texts if t]
        if not texts:
            out[code] = {"label": "", "medoid": "", "n_used": 0}
            continue

        gtr = gtr_embed(texts)                       # [k, 768]
        centroid = gtr.mean(0, keepdim=True)
        centroid = centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-9)
        medoid = gtr[0:1]                             # centroid-nearest doc

        label = vec2text.invert_embeddings(centroid, corrector=corr,
                                           num_steps=NUM_STEPS)[0]
        med = vec2text.invert_embeddings(medoid, corrector=corr,
                                         num_steps=NUM_STEPS)[0]
        out[code] = {"label": label.strip(), "medoid": med.strip(),
                     "n_used": len(texts)}
        print(f"  {code:>6} (k={len(texts)}): {label.strip()[:90]}")

    (DATA / "vec2text_labels.json").write_text(json.dumps(out, indent=2,
                                                          ensure_ascii=False))
    print("wrote vec2text_labels.json")


if __name__ == "__main__":
    main()
