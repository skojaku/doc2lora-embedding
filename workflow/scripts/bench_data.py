"""Where the benchmark data lives, and the two loaders every benchmark shares.

This module is the single place that answers "which file holds the paper table /
the author-paper edges / the citation network / the embeddings for this corpus".
Fourteen scripts across workflow/ and exps/ used to import that answer from
`eval_collab.py`, which also carried a (now removed) evaluation of its own, and
which hardcoded absolute paths like /data/datasets/aps/preprocessed -- so the
scripts only ran on the machine the paper was produced on.

Paths resolve in this order, so the same code works under Snakemake, from a
shell, and on someone else's filesystem:

  1. an environment variable, e.g. $APS_PAPER_TABLE
  2. the matching key in workflow/config.yaml
  3. the built-in default (the layout of the machine the paper was produced on)

Keys are the same ones the Snakefile reads, so there is nothing extra to
configure: set them once in workflow/config.yaml.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# The repository root, found from this file rather than from the process cwd.
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "workflow" / "config.yaml"

# Built-in fallbacks: the layout on the machine the manuscript was produced on.
# They are last resort only -- set the keys in workflow/config.yaml instead.
DEFAULTS = {
    "aps_paper_table": "/data/datasets/aps/preprocessed/paper_table.csv",
    "aps_author_net": "/data/datasets/aps/preprocessed/paper_author_net.npz",
    "aps_citation_net": "/data/datasets/aps/preprocessed/citation_net.npz",
    "fields_prep_base": "/data/projects/gravity-of-ideas/tmp-data/preprocessed",
    "openalex_master_paper_table": "/data/datasets/openalex/preprocessed/paper_table.csv",
    "openalex_prep_base": "/data/datasets/openalex/preprocessed",
    "data_dir": "data/",
    "icae_base_model": "mistralai/Mistral-7B-Instruct-v0.2",
}


@functools.lru_cache(maxsize=1)
def config() -> dict:
    """workflow/config.yaml if it exists, else {} (so a bare checkout still imports)."""
    if not CONFIG.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with CONFIG.open() as fh:
        return yaml.safe_load(fh) or {}


def path(key: str) -> str:
    """Resolve one data path: $ENV_VAR, then workflow/config.yaml, then the default."""
    env = os.environ.get(key.upper())
    if env:
        return env
    val = config().get(key)
    if val:
        return str(val)
    if key not in DEFAULTS:
        raise KeyError(f"no path known for {key!r}; set it in workflow/config.yaml")
    return DEFAULTS[key]


def data_dir() -> str:
    """The workflow's data root, normalised (see the Snakefile: "./data/" != "data/")."""
    return os.path.normpath(path("data_dir"))


def aps_paper_table() -> str:
    return path("aps_paper_table")


def aps_author_net() -> str:
    return path("aps_author_net")


def aps_citation_net() -> str:
    return path("aps_citation_net")


def fields_prep_base() -> str:
    """Directory holding the per-field OpenAlex slices (openalex-<field>/...)."""
    return path("fields_prep_base")


def openalex_master_paper_table() -> str:
    return path("openalex_master_paper_table")


def openalex_prep_base() -> str:
    """Directory holding the full OpenAlex tables (paper_table.csv, abstracts.parquet,
    citation_net.npz) that the general-adapter sampling reads."""
    return path("openalex_prep_base")


def citation_net(field: str) -> str:
    """The citation network for a corpus: APS has its own, fields live under the slices."""
    if field == "aps":
        return aps_citation_net()
    return f"{fields_prep_base()}/openalex-{field}/citation_net.npz"


def load(field: str):
    """Locate one corpus: (paper table with a `year` column, author-paper edges, embedding dir).

    Returns
    -------
    pt : DataFrame  [paper_id, frac_year, year]
    edges : DataFrame  [paper_id, author_id]
    emb_dir : str  directory holding that corpus's embedding npz files
    """
    dd = data_dir()
    if field == "aps":
        pt = pd.read_csv(aps_paper_table(), usecols=["paper_id", "frac_year"]).dropna()
        pt["year"] = pt["frac_year"].astype(float).astype(int)
        d = np.load(aps_author_net(), allow_pickle=True)
        PA = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
        coo = PA.tocoo()
        edges = pd.DataFrame({"paper_id": coo.row, "author_id": coo.col})
        emb_dir = os.path.join(dd, "aps", "embeddings")
    elif field.startswith("arxiv_"):
        b = os.path.join(dd, "fields", field)
        pt = pd.read_csv(f"{b}/paper_table.csv", usecols=["paper_id", "frac_year"]).dropna()
        pt["year"] = pt["frac_year"].astype(float).astype(int)
        edges = (pd.read_csv(f"{b}/author_paper_table.csv", usecols=["paper_id", "author_id"])
                 .dropna().astype(np.int64))
        emb_dir = f"{b}/embeddings"
    else:
        src = f"{fields_prep_base()}/openalex-{field}"
        pt = pd.read_csv(f"{src}/paper_table.csv", usecols=["paper_id", "frac_year"]).dropna()
        pt["year"] = pt["frac_year"].astype(float).astype(int)
        edges = (pd.read_csv(f"{src}/author_paper_table.csv", usecols=["paper_id", "author_id"])
                 .dropna().astype(np.int64))
        emb_dir = os.path.join(dd, "fields", field, "embeddings")
    return pt, edges, emb_dir


def collab_net(ap, n_auth, n_pap):
    """Unweighted author-author co-authorship network from author-paper incidences."""
    a2p = sp.csr_matrix((np.ones(len(ap)), (ap["author_id"], ap["paper_id"])),
                        shape=(n_auth, n_pap))
    net = (a2p @ a2p.T).tocsr()
    net.data[:] = 1
    net.setdiag(0)
    net.eliminate_zeros()
    return net.tocsr()


if __name__ == "__main__":       # `python workflow/scripts/bench_data.py` prints what resolved
    for k in sorted(DEFAULTS):
        src = ("env" if os.environ.get(k.upper()) else
               "config" if config().get(k) else "default")
        print(f"{k:32s} {path(k):60s} [{src}]")


# Where a chain WRITES. The scripts all live in one directory now, so a script can
# no longer take "next to me" to mean "my results": this is the one place that
# answers where a chain's outputs go. Override per chain with $<TOPIC>_OUT, or move
# the whole tree with `data_dir` in workflow/config.yaml.
def out_dir(topic: str) -> Path:
    env = os.environ.get(f"{topic.upper()}_OUT")
    base = Path(env) if env else ROOT / data_dir() / topic
    base.mkdir(parents=True, exist_ok=True)
    return base
