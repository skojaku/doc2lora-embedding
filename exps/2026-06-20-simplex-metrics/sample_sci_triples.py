"""Sample corpus-grounded scientific corner triples (#60 Metric 3).

K-means the APS SPECTER2 embeddings, then greedily carve the cluster centroids
into N disjoint triples each chosen to be MAXIMALLY mutually distant (so every
triple is three far-apart scientific topics, the hard case for fusion). For each
chosen cluster the corner document is the medoid paper (title + abstract nearest
the centroid); an LLM names it and extracts its characteristic words. Emits
corners_sci1.json .. corners_sciN.json in the schema simplex_common reads.

  set -a; . ../../.env; set +a
  python sample_sci_triples.py --n-triples 3 --k 48 --seed 42

Heavy-ish (loads 644k x 768 SPECTER2) + N*3 OpenRouter calls for naming/words.
Fully deterministic given --seed.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
URL = "https://openrouter.ai/api/v1/chat/completions"
LABEL_MODEL = "google/gemini-3.5-flash"


def llm_json(prompt, max_tokens=600):
    body = {"model": LABEL_MODEL, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}
    for _ in range(4):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {OR_KEY}"},
                              json=body, timeout=120)
            if r.status_code == 400 and "response_format" in body:
                body.pop("response_format"); continue
            txt = r.json()["choices"][0]["message"]["content"]
            return json.loads(re.search(r"\{.*\}", txt, flags=re.S).group(0))
        except Exception:
            continue
    return {}


_BADTITLE = {"", "none", "nan", "null", "untitled"}
_STOP = set("the a an of and or to in on for with by from as is are be that this we "
            "our it its their they these those which whose can may using use based "
            "given each both into over under more most than then thus at we present "
            "show study results new two three between within across about".split())


def clean_title(title):
    t = (title or "").strip()
    return "" if t.lower() in _BADTITLE else t


def local_words(text, cap=14):
    """Offline keyword fallback: ordered de-duped content words (no LaTeX glue)."""
    out = []
    for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower()):
        if len(w) > 3 and w not in _STOP and w not in out:
            out.append(w)
        if len(out) >= cap:
            break
    return out


def name_and_words(title, abstract):
    title = clean_title(title)
    body = title or abstract
    out = llm_json(
        "From this physics paper, give a 2-4 word topic NAME and 12-14 short "
        "lowercase characteristic keywords/phrases. Reply ONLY as JSON "
        '{"name": "...", "words": ["...", ...]}.\n\n'
        f'Title: {title}\nAbstract: {abstract[:1500]}')
    name = str(out.get("name", "")).strip()
    if name.lower() in _BADTITLE:
        name = ""
    words = [str(w).strip().lower() for w in out.get("words", []) if str(w).strip()][:14]
    # offline fallbacks so a corner is never left nameless / word-less
    if not name:
        name = title[:60] or " ".join(local_words(abstract)[:4]) or "physics topic"
    if not words:
        words = local_words(f"{title}. {abstract}")
    return name[:60], words


def greedy_triples(centroids, n_triples, rng):
    """Carve cluster indices into n disjoint triples of maximal mutual distance."""
    # pairwise cosine distance between L2-normalised centroids
    C = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - C @ C.T
    avail = set(range(len(centroids)))
    triples = []
    for _ in range(n_triples):
        if len(avail) < 3:
            break
        av = sorted(avail)
        # seed each triple from the single most-distant available pair, then add
        # the available cluster maximising summed distance to the pair.
        best = None
        for a_i in range(len(av)):
            for b_i in range(a_i + 1, len(av)):
                a, b = av[a_i], av[b_i]
                if best is None or D[a, b] > best[0]:
                    best = (D[a, b], a, b)
        _, a, b = best
        rest = [c for c in av if c not in (a, b)]
        c = max(rest, key=lambda x: D[a, x] + D[b, x])
        tri = [a, b, c]
        triples.append(tri)
        avail -= set(tri)
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(HERE, "..", "..",
                    "data/aps/embeddings/baseline_specter2.npz"))
    ap.add_argument("--parquet", default=os.path.join(HERE, "..", "..",
                    "data/aps/paper_text.parquet"))
    ap.add_argument("--n-triples", type=int, default=3)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-abstract", type=int, default=400)
    args = ap.parse_args()

    z = np.load(args.emb)
    vecs, pids = z["vecs"].astype(np.float32), z["paper_ids"]
    df = pd.read_parquet(args.parquet, columns=["aps_paper_id", "title", "abstract"])
    df = df.set_index("aps_paper_id")

    # keep only papers with a substantive abstract (the medoid must be readable)
    have = df["abstract"].fillna("").str.len() >= args.min_abstract
    keep_ids = set(df.index[have])
    mask = np.array([p in keep_ids for p in pids])
    vecs, pids = vecs[mask], pids[mask]
    print(f"{len(pids)} papers with abstract >= {args.min_abstract} chars")

    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3,
                         batch_size=4096)
    lab = km.fit_predict(vecs)
    cents = km.cluster_centers_

    triples = greedy_triples(cents, args.n_triples, np.random.default_rng(args.seed))
    print(f"selected {len(triples)} disjoint triples from {args.k} clusters")

    for t_idx, tri in enumerate(triples, 1):
        spec = {"set": f"sci{t_idx}", "keys": ["A", "B", "C"], "corners": {}}
        for code, cl in zip(["A", "B", "C"], tri):
            members = np.where(lab == cl)[0]
            cv = cents[cl] / (np.linalg.norm(cents[cl]) + 1e-12)
            mv = vecs[members] / (np.linalg.norm(vecs[members], axis=1, keepdims=True) + 1e-12)
            medoid = members[int(np.argmax(mv @ cv))]
            pid = pids[medoid]
            row = df.loc[pid]
            title, abstract = str(row["title"]), str(row["abstract"])
            name, words = name_and_words(title, abstract)
            ct = clean_title(title)
            lead = (f"{ct}. {abstract}" if ct else abstract).strip()
            spec["corners"][code] = {"name": name, "paper_id": int(pid),
                                     "cluster": int(cl), "lead": lead, "words": words}
            print(f"  sci{t_idx}.{code}  cl{cl:>2}  {name}  (pid {pid})")
        out = os.path.join(HERE, f"corners_sci{t_idx}.json")
        json.dump(spec, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"wrote {out}\n")


if __name__ == "__main__":
    main()
