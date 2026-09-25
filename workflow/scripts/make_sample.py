"""Build the bundled sample corpus: a small synthetic field the real rules can run on.

WHY A SYNTHETIC CORPUS. The reported results need a licensed APS corpus, an OpenAlex
snapshot, and three hypernetwork checkpoints on a GPU. None of that belongs in a test
you can run while reading the code. What a test CAN do is take the half of the
workflow that turns representations into reported numbers -- score pools, bootstrap
intervals, the benchmark table, the label metrics, the judge panel -- and run the real
scripts over vectors whose answer is known in advance.

WHAT IS PLANTED. Every paper gets a latent topic vector. Each "method" then sees that
latent through its own noise level, so the ordering of the methods is fixed before any
scoring happens:

    genkron < sbert < kron < gte < emb.gemma < specter2 < instructor < raw gene
    (lower noise, better scores) ----------------------------------> (worst)

`check_sample.py` recomputes that ordering from what the workflow produced and fails if
it disagrees. A pipeline that silently drops a method, mixes up a column, or loses the
pairing between methods cannot pass, because those failures all move the ordering.

The authorship graph is generated, not planted: authors have home topics and drift
between them, and the collaboration benchmark is then built from it by the same
`save_collab_scores.py` the reported numbers use.

WHAT IT DOES NOT COVER. Gene extraction, decoding, and anything else that needs a
checkpoint. Those rules are named in REPRODUCE.md and are the part a sample corpus
cannot stand in for.

    python workflow/scripts/make_sample.py --out-prep data/sample/prep
"""
import argparse
import json
import zlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import data_dir, out_dir  # noqa: E402

FIELD = "sample"
N_TOPICS = 10
N_PAPERS = 3000
N_AUTHORS = 400
DIM = 64
YEARS = list(range(2004, 2020))
AUTHORS_PER_PAPER = (1, 4)
TOPIC_COS = 0.72       # how alike two topics of the same field are. Near-orthogonal
                       # topics make classification trivial at every noise level, and
                       # a saturated task cannot rank the methods.
SEED = 20260921

# Noise each method adds to the latent topic vector, as a fraction of the signal's
# own length. This IS the expected ranking. Noise is scaled by 1/sqrt(DIM) so the
# numbers mean what they look like: in 64 dimensions an unscaled N(0, s) vector is
# ~8s long, which buries a unit-length signal at any s worth writing down.
METHOD_NOISE = {
    "qwen_genkron_emb.npz": ("embeddings", 0.35),
    "sbert_allmpnet.npz": ("vecs", 0.45),
    "qwen_kron_emb.npz": ("embeddings", 0.55),
    "baseline_gte.npz": ("vecs", 0.70),
    "baseline_embeddinggemma.npz": ("vecs", 0.90),
    "baseline_specter2.npz": ("vecs", 1.15),
    "baseline_instructor.npz": ("vecs", 1.45),
    "qwen_norm_lora_emb.npz": ("embeddings", 1.80),
    # the two other encoders exist only so save_collab_scores.py scores its full roster
    "gemma_norm_lora_emb.npz": ("embeddings", 2.10),
    "mistral_norm_lora_emb.npz": ("embeddings", 1.70),
}

TOPIC_NAMES = [
    "Quantum information",
    "Condensed matter physics",
    "Astrophysics and cosmology",
    "Fluid dynamics",
    "Particle physics",
    "Statistical mechanics",
    "Nonlinear optics",
    "Plasma physics",
    "Superconductivity",
    "Atomic and molecular physics",
]

# Per-method label quality, mirroring what the reported comparison found: the gene
# decode names the field, ICAE and the in-context reader land close, KeyLLM returns
# keywords, vec2text returns noise, and T2L repeats one label everywhere.
LABEL_STYLE = {
    "doc2lora": lambda name, rng: name,
    "icae": lambda name, rng: name.split()[0] + " physics",
    "incontext": lambda name, rng: name.rsplit(" ", 1)[0] if " " in name else name,
    "keyllm": lambda name, rng: ", ".join(
        [w.lower() for w in name.split() if len(w) > 3] + ["lattice", "measurement"]),
    "vec2text": lambda name, rng: "in-distance " + "".join(rng.choice(list("aeilnrst"), 9)) + " scattering",
    "t2l_gte": lambda name, rng: "Social psychology",
    "t2l_hidden": lambda name, rng: "Social psychology",
    "t2l_dw": lambda name, rng: "Social psychology",
    # BERTopic returns a c-TF-IDF keyword list, and the `3` arms truncate the same
    # ranking to three words -- the length-matched control.
    "bertopic": lambda name, rng: ", ".join(
        [w.lower() for w in name.replace(",", "").split()] + ["lattice", "phase", "model"]),
    "bertopic3": lambda name, rng: " ".join(
        [w.lower() for w in name.replace(",", "").split()][:3]),
    "bertopic_kbi": lambda name, rng: ", ".join(
        [w.lower() for w in name.replace(",", "").split()][::-1] + ["coupling", "regime"]),
    "bertopic_kbi3": lambda name, rng: " ".join(
        [w.lower() for w in name.replace(",", "").split()][::-1][:3]),
}


def build(out_prep: Path, out_data: Path):
    rng = np.random.default_rng(SEED)

    # ── latent structure ─────────────────────────────────────────────────
    # Topics of ONE field, not of unrelated sciences: they share a large common
    # component, so their centroids sit at cosine ~.6 rather than near-orthogonal.
    # Random centroids in 64 dimensions are almost orthogonal, which makes the
    # classification task trivial at every noise level and erases the ordering the
    # test is supposed to recover.
    common = rng.normal(size=DIM)
    common /= np.linalg.norm(common)
    alpha = np.sqrt(1.0 / TOPIC_COS - 1.0)      # <c_i, c_j> ~ 1 / (1 + alpha^2)
    unique = rng.normal(size=(N_TOPICS, DIM))
    unique /= np.linalg.norm(unique, axis=1, keepdims=True)
    centroids = common + alpha * unique
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    topic = rng.integers(0, N_TOPICS, N_PAPERS)
    scale = 1.0 / np.sqrt(DIM)          # so a noise level reads as a fraction of the signal
    latent = centroids[topic] + 0.30 * scale * rng.normal(size=(N_PAPERS, DIM))
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    paper_ids = np.arange(N_PAPERS, dtype=np.int64)
    year = rng.choice(YEARS, N_PAPERS)

    # ── authorship: home topics, with drift, so co-authors share a subject ──
    home = rng.integers(0, N_TOPICS, N_AUTHORS)
    by_topic = {t: paper_ids[topic == t] for t in range(N_TOPICS)}
    rows = []
    for a in range(N_AUTHORS):
        n_pap = int(rng.integers(12, 40))
        own = rng.choice(by_topic[home[a]], size=min(n_pap, len(by_topic[home[a]])), replace=False)
        drift_topic = int(rng.integers(0, N_TOPICS))
        drift = rng.choice(by_topic[drift_topic], size=max(1, n_pap // 5), replace=False)
        for p in np.concatenate([own, drift]):
            rows.append((int(p), a))
    ap = pd.DataFrame(rows, columns=["paper_id", "author_id"]).drop_duplicates()

    # keep papers to a believable author count, so co-authorship is not a clique
    keep = []
    for pid, g in ap.groupby("paper_id"):
        cap = int(rng.integers(*AUTHORS_PER_PAPER)) + 1
        keep.append(g.sample(n=min(cap, len(g)), random_state=int(pid)))
    ap = pd.concat(keep, ignore_index=True).sort_values(["paper_id", "author_id"])

    # ── the tables bench_data.load() reads for an OpenAlex-style field ────
    src = out_prep / f"openalex-{FIELD}"
    src.mkdir(parents=True, exist_ok=True)
    pt = pd.DataFrame({"paper_id": paper_ids, "frac_year": year.astype(float)})
    pt.to_csv(src / "paper_table.csv", index=False)
    ap.to_csv(src / "author_paper_table.csv", index=False)

    # ── topic labels for the classification task ─────────────────────────
    fdir = out_data / "fields" / FIELD
    (fdir / "embeddings").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"paper_id": paper_ids,
                  "main_class": topic.astype(int),
                  "sub_class": (topic * 10 + rng.integers(0, 3, N_PAPERS)).astype(int)},
                 ).to_parquet(fdir / "paper_topics.parquet", index=False)

    # ── one embedding file per method, at its planted noise level ────────
    for fn, (key, sigma) in METHOD_NOISE.items():
        # a stable per-method seed: Python's hash() is salted per process
        noise_rng = np.random.default_rng(zlib.crc32(fn.encode()))
        E = latent + sigma * scale * noise_rng.normal(size=(N_PAPERS, DIM))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        np.savez_compressed(fdir / "embeddings" / fn,
                            paper_ids=paper_ids, **{key: E.astype(np.float32)})

    # ── label-evaluation nodes, one per topic plus two sub-nodes each ─────
    lrng = np.random.default_rng(SEED + 1)
    nodes = []
    for t, name in enumerate(TOPIC_NAMES):
        for k in range(3):
            code = f"{t}" if k == 0 else f"{t}.{k}"
            gt = name if k == 0 else f"{name}, topic {k}"
            nodes.append({
                "code": code,
                "kind": "field" if k == 0 else "subdivision",
                "gt_label": gt,
                "decoded": {m: f(gt, lrng) for m, f in LABEL_STYLE.items()},
            })
    ldir = out_dir("labels")   # honours $LABELS_OUT, so a test writes off the real tree
    (ldir / "label_eval_nodes.json").write_text(json.dumps(nodes, indent=2))

    manifest = {
        "field": FIELD, "seed": SEED, "papers": N_PAPERS, "authors": N_AUTHORS,
        "author_paper_rows": int(len(ap)), "topics": N_TOPICS, "dim": DIM,
        "label_nodes": len(nodes),
        "planted_order": [fn for fn, _ in sorted(METHOD_NOISE.items(), key=lambda kv: kv[1][1])],
        "method_noise": {fn: s for fn, (_, s) in METHOD_NOISE.items()},
    }
    (out_data / "sample" / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (out_data / "sample" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"sample corpus: {N_PAPERS} papers, {N_AUTHORS} authors, "
          f"{len(ap)} author-paper rows, {len(nodes)} label nodes")
    print(f"  tables      {src}")
    print(f"  embeddings  {fdir / 'embeddings'} ({len(METHOD_NOISE)} methods)")
    print(f"  labels      {ldir / 'label_eval_nodes.json'}")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out-prep", default=None, help="where the openalex-<field> tables go")
    ap_.add_argument("--out-data", default=None, help="the workflow's data root")
    a = ap_.parse_args()
    data = Path(a.out_data) if a.out_data else Path(data_dir())
    build(Path(a.out_prep) if a.out_prep else data / "sample" / "prep", data)
