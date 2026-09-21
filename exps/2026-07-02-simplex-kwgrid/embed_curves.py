"""Robustness of the fusion-response curve to the choice of embedding model.

Re-embed the ALREADY-decoded abstracts (no re-decode) with a chosen backend, remeasure
each cell's corner weights w = softmax(cos(decoded_abs, source_abs_j)/tau0) with a FIXED
tau0 (no per-method optimization), and dump the pooled (ideal, measured) points per method.
plot_embcurves.py overlays all backends to check the picture is model-independent.

Backends (run each with its env's python):
  mpnet   : this env  -> all-mpnet-base-v2                       (SBERT, default)
  gtr     : this env  -> sentence-transformers/gtr-t5-base       (ST-wrapped GTR)
  specter2: .venv-specter/bin/python  -> allenai/specter2 (adapters, CLS)
  gemma   : .venv-emgemma/bin/python  -> google/embeddinggemma-300m

  python embed_curves.py mpnet
  python embed_curves.py gtr
  ../../.venv-specter/bin/python embed_curves.py specter2
  ../../.venv-emgemma/bin/python embed_curves.py gemma
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_common as sc

TAU = 0.30
METHODS = ["doc2lora", "incontext", "icae"]
SUF = os.environ.get("KWTAG", "_kwsrc")
CENTER = os.environ.get("CENTER", "1") == "1"   # subtract the 3-corner centroid before cosine


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def get_encoder(backend):
    if backend in ("mpnet", "gtr"):
        from sentence_transformers import SentenceTransformer
        name = "all-mpnet-base-v2" if backend == "mpnet" else "sentence-transformers/gtr-t5-base"
        m = SentenceTransformer(name, device="cuda")
        return lambda texts: unit(np.asarray(m.encode(texts, normalize_embeddings=True,
                                                       batch_size=64, show_progress_bar=False)))
    if backend == "gemma":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("google/embeddinggemma-300m", device="cuda")
        return lambda texts: unit(np.asarray(m.encode(texts, normalize_embeddings=True,
                                                       batch_size=32, show_progress_bar=False)))
    if backend == "specter2":
        import torch
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        mdl = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        ad = mdl.load_adapter("allenai/specter2", source="hf", load_as="prox", set_active=True)
        mdl.set_active_adapters(ad); mdl.to(dev).eval()

        def enc(texts):
            out = []
            for i in range(0, len(texts), 32):
                b = texts[i:i + 32]
                t = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
                with torch.no_grad():
                    o = mdl(**t)
                out.append(o.last_hidden_state[:, 0, :].cpu().numpy())   # CLS
            return unit(np.vstack(out))
        return enc
    raise SystemExit(f"unknown backend {backend}")


def resp(cos):
    e = np.exp((cos - cos.max()) / TAU)
    return e / e.sum()


def main(backend):
    man = {m["set"]: m for m in json.load(open(os.path.join(sc.HERE, "mild_manifest.json")))}
    sets = [f"{s}_mild" for s in man
            if os.path.exists(os.path.join(sc.RESULTS, f"absfollow_{s}_mild{SUF}.json"))]
    print(f"{backend}: {len(sets)} sets", flush=True)

    # gather every text once (source leads + decoded abstracts), dedup, encode in one pass
    texts, idx = [], {}
    def tid(t):
        if t not in idx:
            idx[t] = len(texts); texts.append(t)
        return idx[t]

    records = []   # (method, ideal_vec, decoded_id, [srcA,srcB,srcC ids])
    for s in sets:
        spec = sc.load_corners(sc.corners_path(s))
        src_ids = [tid(spec["leads"][k]) for k in spec["keys"]]
        base = json.load(open(os.path.join(sc.RESULTS, f"absfollow_{s}{SUF}.json")))["cells"]
        icp = os.path.join(sc.RESULTS, f"absfollow_{s}{SUF}_icae.json")
        icae = {tuple(c["bary"]): c.get("icae", {}) for c in json.load(open(icp))["cells"]} if os.path.exists(icp) else {}
        for c in base:
            b = tuple(c["bary"]); tot = sum(b); ideal = [x / tot for x in b]
            srcs = {"doc2lora": c.get("doc2lora", {}), "incontext": c.get("incontext", {}),
                    "icae": icae.get(b, {})}
            for ch in METHODS:
                ab = (srcs[ch] or {}).get("abs", "")
                if len(ab.split()) < 8:
                    continue
                records.append((ch, ideal, tid(ab), src_ids))

    print(f"encoding {len(texts)} unique texts ...", flush=True)
    E = get_encoder(backend)(texts)

    pts = {ch: {"x": [], "y": []} for ch in METHODS}
    for ch, ideal, did, sids in records:
        S = E[sids]                                  # 3 x d
        if CENTER:
            c = S.mean(0)
            dvec = unit(E[did] - c); Sv = unit(S - c)   # de-mean by corner centroid, re-unit
        else:
            dvec = E[did]; Sv = S
        cos = dvec @ Sv.T
        w = resp(cos)
        for j in range(3):
            pts[ch]["x"].append(float(ideal[j])); pts[ch]["y"].append(float(w[j]))
    out = os.path.join(sc.RESULTS, f"embcurve_{backend}.json")
    json.dump({"backend": backend, "tau": TAU, "n_records": len(records), "pts": pts}, open(out, "w"))
    print(f"wrote {out}  ({len(records)} cell-method records)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
