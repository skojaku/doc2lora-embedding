"""BERTopic baseline for the PACS cluster-labeling evaluation (issue #24).

BERTopic is the standard "name a cluster of documents" pipeline: embed ->
UMAP -> HDBSCAN -> c-TF-IDF keywords per cluster. Here the clusters are GIVEN
(the PACS nodes the whole evaluation is built on), so we use BERTopic's
documented *manual topic modeling* mode -- ``BaseDimensionalityReduction`` +
``BaseCluster`` with the cluster assignment handed in as ``y`` -- which skips
UMAP/HDBSCAN and runs exactly the part of BERTopic under test: the topic
REPRESENTATION (c-TF-IDF, optionally re-ranked by KeyBERTInspired).

Why one fit per level. c-TF-IDF is contrastive: a word scores high for a cluster
because it is frequent there and rare in the *other* clusters of the same model.
The PACS hierarchy is nested, so a document belongs to one node per level and a
single flat fit cannot hold all three levels at once. We therefore fit three
models -- main / division / subdivision -- each over EVERY PACS group at that
level with >= MIN_DOCS papers (9 / 65 / 393 groups), which is the most generous
contrast set available, and then read off the 28 evaluated nodes.

Fairness to the other methods:
  - same member pool (paper_groups.parquet), same CAP=2000 papers per node and
    same seed 0 as compute_qwen_fullrank_means.py (Doc2LoRA's full-rank means),
  - same document string (paper_text.parquet ``text`` = title + abstract; the
    "Title:"/"Abstract:" scaffold is stripped since this is a bag-of-words
    method). ~31% of APS abstracts carry LaTeX / MathML markup
    (``\\ensuremath``, entity-encoded ``<mi>``/``<msup>``), which an LLM reader
    ignores but c-TF-IDF happily ranks as topic words, so by default the markup
    is stripped before vectorising. Pass ``raw`` as the third argument to skip
    that and write the ``bertopic_raw_*`` arms instead, which measures how much
    of BERTopic's score is preprocessing rather than method.
  - papers are restricted to those that also have an all-mpnet SBERT vector
    (~96% of the corpus) so the vanilla and KeyBERTInspired arms see exactly the
    same documents. The SBERT vectors are handed to BERTopic as precomputed
    document embeddings, so no re-encoding happens.
  - native output only: the keyword list IS the label (no naming LLM), exactly
    as for the KeyLLM baseline.

Deviation from BERTopic's defaults, in BERTopic's favour and declared in the
output meta: English stop words are removed (the default CountVectorizer keeps
them). ``min_df`` is left at 1 -- BERTopic fits the vectorizer on the per-TOPIC
concatenated text, so a document-frequency floor would be a floor on the number
of *clusters* a word must appear in, i.e. exactly the distinctive words c-TF-IDF
is looking for.

Arms written (``top_n_words=10`` is BERTopic's default; the 3-word truncations
are LENGTH-MATCHED controls for Doc2LoRA's ~2.4-word labels, since the reported
fuzzy metric penalises long strings):
  bertopic_labels.json       c-TF-IDF, top 10
  bertopic3_labels.json      c-TF-IDF, top 3
  bertopic_kbi_labels.json   KeyBERTInspired, top 10
  bertopic_kbi3_labels.json  KeyBERTInspired, top 3
  bertopic_topics.json       full ranked word lists + scores + per-level meta

Run:  .venv-bertopic/bin/python bertopic_labels.py [CAP] [MIN_DOCS] [clean|raw]
      (CUDA_VISIBLE_DEVICES=<gpu> speeds up the KeyBERTInspired word embedding;
       CPU works too.)
"""

import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

ROOT = HERE.parent.parent
CA = ROOT / "data/pacs/results"

PAPER_GROUPS = CA / "paper_groups.parquet"
PAPER_TEXT = ROOT / "data/aps/paper_text.parquet"
SBERT = ROOT / "data/aps/embeddings/sbert_allmpnet.npz"
NODES = DATA / "nodes.json"

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000   # papers per node (doc2lora's CAP)
MIN_DOCS = int(sys.argv[2]) if len(sys.argv) > 2 else 100   # groups kept in the fit
MODE = sys.argv[3] if len(sys.argv) > 3 else "clean"        # "clean" | "raw" markup handling
assert MODE in ("clean", "raw")
TAG = "" if MODE == "clean" else "_raw"
SEED = 0
TOP_N = 10          # BERTopic's default top_n_words
SHORT_N = 3         # length-matched control
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"

# experiment field digit -> PACS main_class_id is the identity (see label_eval_prep)
LEVEL_COL = {"main": "main_class_id", "division": "division", "subdivision": "subdivision"}

OUT_TOPICS = DATA / f"bertopic{TAG}_topics.json"
OUT = {
    ("ctfidf", TOP_N): DATA / f"bertopic{TAG}_labels.json",
    ("ctfidf", SHORT_N): DATA / f"bertopic{TAG}3_labels.json",
    ("keybert", TOP_N): DATA / f"bertopic{TAG}_kbi_labels.json",
    ("keybert", SHORT_N): DATA / f"bertopic{TAG}_kbi3_labels.json",
}

# LaTeX / MathML strippers (MODE="clean"): entity-decode, drop tags, drop inline
# math spans, drop backslash commands, drop the leftover brace/sub/superscript
# punctuation. Applied before the CountVectorizer sees the text.
_TAG_RE = re.compile(r"<[^>]{0,200}>")
_MATH_RE = re.compile(r"\$\$?[^$]{0,400}\$\$?")
_CMD_RE = re.compile(r"\\[A-Za-z]+|\\[^A-Za-z\s]")
_PUNCT_RE = re.compile(r"[{}$^_~]")


def eval_nodes():
    """(code, level, group value) for the 28 scored nodes, from nodes.json."""
    tree = json.loads(NODES.read_text())
    out = []
    for f in tree["children"]:
        out.append((str(f["fid"]), "main", int(f["fid"])))
        for d in f["children"]:
            out.append((d["ref"], "division", d["ref"]))
            for s in d["children"]:
                out.append((s["ref"], "subdivision", s["ref"]))
    return out


def prepare(t):
    """paper_text stores 'Title: ...\\nAbstract: ...'; drop the field markers, and
    (MODE="clean") the LaTeX / MathML markup a bag-of-words model would rank."""
    t = t.replace("Title:", " ").replace("Abstract:", " ")
    if MODE == "raw":
        return t
    t = html.unescape(html.unescape(t))
    t = _TAG_RE.sub(" ", t)
    t = _MATH_RE.sub(" ", t)
    t = _CMD_RE.sub(" ", t)
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def build_level(pg, id2text, emb_row, level, wanted):
    """Documents + cluster ids for every group at `level` with >= MIN_DOCS papers."""
    key = pg[LEVEL_COL[level]].astype(str)
    vc = key.value_counts()
    groups = sorted(str(g) for g in vc[vc >= MIN_DOCS].index if g not in ("<NA>", "nan", "None"))
    groups = sorted(set(groups) | {str(w) for w in wanted})   # never drop a scored node
    members = {g: v.to_numpy() for g, v in pg.paper_id.groupby(key)}
    rng = np.random.default_rng(SEED)

    docs, ys, rows, sizes = [], [], [], {}
    for gi, g in enumerate(groups):
        pids = np.array([int(p) for p in members.get(g, ())
                         if int(p) in id2text and int(p) in emb_row])
        if len(pids) == 0:
            continue
        sel = pids if len(pids) <= CAP else rng.choice(pids, size=CAP, replace=False)
        docs.extend(prepare(id2text[int(p)]) for p in sel)
        ys.extend([gi] * len(sel))
        rows.extend(emb_row[int(p)] for p in sel)
        sizes[g] = {"n_used": int(len(sel)), "n_total": int(len(pids))}
    return groups, docs, np.asarray(ys), np.asarray(rows), sizes


def fit_level(docs, ys, embeddings):
    """Manual-mode BERTopic: predefined clusters, c-TF-IDF + KeyBERTInspired."""
    from bertopic import BERTopic
    from bertopic.cluster import BaseCluster
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.representation import KeyBERTInspired
    from sklearn.feature_extraction.text import CountVectorizer

    model = BERTopic(
        embedding_model=EMB_MODEL,            # only used to embed candidate WORDS
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(),
        vectorizer_model=CountVectorizer(stop_words="english"),
        representation_model={"KeyBERT": KeyBERTInspired(top_n_words=TOP_N)},
        top_n_words=TOP_N,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings, y=list(ys))
    topics = np.asarray(topics)

    # BERTopic re-numbers topics by size, so recover y -> topic id from the
    # aligned assignment and assert the mapping is a bijection.
    mapping = {}
    for y, t in zip(ys, topics):
        mapping.setdefault(int(y), set()).add(int(t))
    bad = {y: sorted(t) for y, t in mapping.items() if len(t) != 1}
    assert not bad, f"cluster ids were not preserved: {bad}"
    mapping = {y: t.pop() for y, t in mapping.items()}
    assert len(set(mapping.values())) == len(mapping), "topic ids collided"
    return model, mapping


def main():
    print(f"CAP={CAP}  MIN_DOCS={MIN_DOCS}  seed={SEED}  markup={MODE}")
    pg = pd.read_parquet(PAPER_GROUPS)
    txt = pd.read_parquet(PAPER_TEXT, columns=["aps_paper_id", "text"])
    id2text = dict(zip(txt["aps_paper_id"].astype(int), txt["text"].astype(str)))
    z = np.load(SBERT)
    emb_all = z["vecs"].astype(np.float32)
    emb_row = {int(p): i for i, p in enumerate(z["paper_ids"].astype(np.int64))}
    print(f"  {len(pg)} assignments | {len(id2text)} texts | {len(emb_row)} sbert vectors")

    wanted = eval_nodes()
    by_level = {}
    for code, level, val in wanted:
        by_level.setdefault(level, []).append((code, str(val)))

    labels = {k: {} for k in OUT}
    record, meta = {}, {}
    for level in ("main", "division", "subdivision"):
        want = by_level[level]
        groups, docs, ys, rows, sizes = build_level(
            pg, id2text, emb_row, level, [v for _, v in want])
        print(f"\n[{level}] {len(groups)} groups | {len(docs)} documents")
        model, mapping = fit_level(docs, ys, emb_all[rows])

        meta[level] = {"n_groups": len(groups), "n_docs": len(docs),
                       "groups": groups, "sizes": sizes}
        for code, val in want:
            tid = mapping[groups.index(val)]
            words = {"ctfidf": model.get_topic(tid),
                     "keybert": model.topic_aspects_["KeyBERT"][tid]}
            record[code] = {"level": level, "group": val,
                            "n_used": sizes[val]["n_used"],
                            "n_total": sizes[val]["n_total"],
                            "ctfidf": [[w, round(float(s), 5)] for w, s in words["ctfidf"]],
                            "keybert": [[w, round(float(s), 5)] for w, s in words["keybert"]]}
            for rep, n in OUT:
                labels[(rep, n)][code] = ", ".join(w for w, _ in words[rep][:n])
            print(f"  {code:>6} | {labels[('ctfidf', TOP_N)][code]}")
        del model, docs, rows

    for (rep, n), path in OUT.items():
        path.write_text(json.dumps(labels[(rep, n)], indent=2, ensure_ascii=False))
    OUT_TOPICS.write_text(json.dumps(
        {"meta": {"cap": CAP, "min_docs": MIN_DOCS, "seed": SEED, "markup": MODE,
                  "top_n_words": TOP_N, "short_n": SHORT_N,
                  "embedding_model": EMB_MODEL,
                  "vectorizer": "CountVectorizer(stop_words='english')",
                  "mode": "manual topic modeling (BaseDimensionalityReduction + BaseCluster, y=PACS node)",
                  "levels": meta},
         "nodes": record}, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_TOPICS.name} + {len(OUT)} label files ({len(record)} nodes)")


if __name__ == "__main__":
    main()
