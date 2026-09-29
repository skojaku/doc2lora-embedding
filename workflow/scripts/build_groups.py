"""Step 1 — assign APS papers to a 4-level concept hierarchy and attach
canonical ground-truth label text.

Levels (broad -> specific):
    main         9 APS subject classes (human titles from category_table.csv)
    division     PACS 2-digit, e.g. "74"        -> pacs_scheme.DIVISIONS
    subdivision  PACS "XX.YY", e.g. "74.20"
    specific     full PACS1 code, e.g. "74.20.-z"

The PACS levels nest definitionally by code prefix (specific < subdivision < division).
Each division is additionally linked to its majority APS main class for a readable top.

Outputs (under results/):
    paper_groups.parquet  one row per paper: paper_id, main_class_id, main_title,
                          division, subdivision, specific
    groups.parquet        one row per group: level, group_id, parent_id,
                          gt_label_text, n_papers
    tree.json             nested main -> division -> subdivision -> specific

Usage (inside the doc2lora container, cwd = /workspace):
    python workflow/scripts/build_groups.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pacs_scheme


def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    exp_dir = Path(__file__).resolve().parents[1]
    results_dir = exp_dir / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    min_group_size = cfg["min_group_size"]
    min_text_length = cfg["min_text_length"]

    # --- available embedding ids ---------------------------------------------
    print("Loading embedding paper_ids...")
    npz = np.load(cfg["mean_pooled_embeddings"], mmap_mode="r")
    emb_ids = set(int(x) for x in npz["paper_ids"])
    print(f"  {len(emb_ids):,} papers have embeddings")

    # --- text-length filter ---------------------------------------------------
    print("Loading paper_text for length filter...")
    txt = pd.read_parquet(cfg["paper_text"], columns=["aps_paper_id", "title", "abstract", "text"])
    txt["text_len"] = txt["text"].astype(str).str.len()
    long_ids = set(txt.loc[txt["text_len"] >= min_text_length, "aps_paper_id"].astype(int))
    print(f"  {len(long_ids):,} papers pass min_text_length={min_text_length}")

    # --- PACS codes -----------------------------------------------------------
    print("Loading paper_table (PACS1)...")
    pt = pd.read_csv(cfg["paper_table"], usecols=["paper_id", "PACS1", "journal_code"], low_memory=False)
    pt = pt[pt["PACS1"].notna() & (pt["PACS1"].astype(str).str.lower() != "none")]
    pt["paper_id"] = pt["paper_id"].astype(int)

    # --- APS main class -------------------------------------------------------
    print("Loading category tables (main class)...")
    cat = pd.read_csv(cfg["category_table"])
    main_titles = dict(
        zip(cat.loc[cat["type"] == "main", "class_id"].astype(int),
            cat.loc[cat["type"] == "main", "title"])
    )
    pct = pd.read_csv(cfg["paper_category_table"])
    # primary assignment per paper (sequence 0 if present, else first)
    pct = pct.sort_values(["paper_id", "sequence"]).drop_duplicates("paper_id", keep="first")
    id_to_main = dict(zip(pct["paper_id"].astype(int), pct["main_class_id"].astype(int)))

    # --- assemble per-paper wide table ---------------------------------------
    valid = emb_ids & long_ids
    pt = pt[pt["paper_id"].isin(valid)].copy()
    print(f"  {len(pt):,} papers with PACS1 + embedding + long text")

    div, sub = zip(*pt["PACS1"].map(pacs_scheme.parse))
    pt["division"] = list(div)
    pt["subdivision"] = list(sub)
    pt["specific"] = pt["PACS1"].astype(str)
    pt = pt[pt["division"].notna()].copy()
    pt["main_class_id"] = pt["paper_id"].map(id_to_main).astype("Int64")  # clean int ids ("3" not "3.0")
    pt["main_title"] = pt["main_class_id"].map(lambda c: main_titles.get(int(c)) if pd.notna(c) else None)

    paper_groups = pt[
        ["paper_id", "main_class_id", "main_title", "division", "subdivision", "specific"]
    ].reset_index(drop=True)
    paper_groups.to_parquet(results_dir / "paper_groups.parquet", index=False)
    print(f"Saved paper_groups.parquet ({len(paper_groups):,} rows)")

    # --- division -> majority main class (for readable top of the tree) -------
    div_to_main = {}
    for d, grp in paper_groups.dropna(subset=["main_class_id"]).groupby("division"):
        div_to_main[d] = int(grp["main_class_id"].mode().iloc[0])

    # --- group summaries ------------------------------------------------------
    rows = []

    def add_groups(level, key_col, label_fn, parent_fn):
        counts = paper_groups[key_col].value_counts()
        for gid, n in counts.items():
            if n < min_group_size:
                continue
            rows.append({
                "level": level,
                "group_id": str(gid),
                "parent_id": parent_fn(gid),
                "gt_label_text": label_fn(gid),
                "n_papers": int(n),
            })

    # main
    add_groups(
        "main", "main_class_id",
        label_fn=lambda c: main_titles.get(int(c), f"class {c}"),
        parent_fn=lambda c: None,
    )
    # division (parent = majority main title)
    add_groups(
        "division", "division",
        label_fn=lambda d: pacs_scheme.describe(d, "division"),
        parent_fn=lambda d: (f"main:{div_to_main[d]}" if d in div_to_main else None),
    )
    # subdivision (parent = division)
    add_groups(
        "subdivision", "subdivision",
        label_fn=lambda s: pacs_scheme.describe(s, "subdivision"),
        parent_fn=lambda s: f"division:{pacs_scheme.parse(s)[0]}",
    )
    # specific (parent = subdivision if present else division)
    def specific_parent(code):
        d, s = pacs_scheme.parse(code)
        return f"subdivision:{s}" if s else f"division:{d}"
    add_groups(
        "specific", "specific",
        label_fn=lambda c: pacs_scheme.describe(c, "specific"),
        parent_fn=specific_parent,
    )

    groups = pd.DataFrame(rows)
    groups.to_parquet(results_dir / "groups.parquet", index=False)
    print(f"Saved groups.parquet")
    print(groups.groupby("level")["group_id"].count().rename("n_groups").to_string())

    # --- tree.json ------------------------------------------------------------
    kept = {lvl: set(groups.loc[groups.level == lvl, "group_id"]) for lvl in groups.level.unique()}

    def children_of(parent_tag, level):
        return sorted(groups.loc[(groups.level == level) & (groups.parent_id == parent_tag), "group_id"])

    tree = {"main": {}}
    for mid, title in main_titles.items():
        if str(mid) not in kept.get("main", set()):
            continue
        node = {"label": title, "divisions": {}}
        for d in [dd for dd, mm in div_to_main.items() if mm == mid and dd in kept.get("division", set())]:
            dnode = {"label": pacs_scheme.describe(d, "division"), "subdivisions": {}}
            for s in children_of(f"division:{d}", "subdivision"):
                snode = {
                    "label": pacs_scheme.describe(s, "subdivision"),
                    "specifics": children_of(f"subdivision:{s}", "specific"),
                }
                dnode["subdivisions"][s] = snode
            node["divisions"][d] = dnode
        tree["main"][str(mid)] = node

    with open(results_dir / "tree.json", "w") as f:
        json.dump(tree, f, indent=2)
    print("Saved tree.json")

    # --- sanity print ---------------------------------------------------------
    print("\n--- sample groups per level ---")
    for lvl in ["main", "division", "subdivision", "specific"]:
        sub = groups[groups.level == lvl].sort_values("n_papers", ascending=False).head(5)
        print(f"\n[{lvl}]")
        for _, r in sub.iterrows():
            print(f"  {r.group_id:<12} n={r.n_papers:<6} parent={str(r.parent_id):<14} {r.gt_label_text}")


if __name__ == "__main__":
    main()
