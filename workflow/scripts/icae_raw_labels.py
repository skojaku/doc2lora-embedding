"""Raw ICAE labels for the cluster-label evaluation.

ICAE answers the field prompt in its OWN decoder (icae_label.py: compress the K
centroid-nearest members into memory slots, average the slots, decode under
shared_prompt.FIELD23_PROMPT -- verbatim the Doc2LoRA instruction and the same
"2 to 3 words" budget). That decoder output is what the evaluation scores; no
naming LLM is applied to it, matching Doc2LoRA, KeyLLM and vec2text, which are
likewise scored on what they natively produce.

Source, in order of preference:
  1. icae_tree_field23.json -- the matched-prompt run (GPU):
       CUDA_VISIBLE_DEVICES=0 python icae_label.py nodes.json icae_tree_field23.json
  2. icae_tree.json -- a local run under whatever prompt icae_label.py carried.
  3. LEGACY: the archived icae_tree.json in git history (TREE_COMMIT), which used
     the older LABEL_PROMPT and its looser "two to five words" budget. Only for
     reference -- it is NOT comparable to Doc2LoRA on a length-sensitive metric,
     and the script says so loudly when it falls back to it.

Output: icae_raw.json   {node_code: raw decoder output}
Run:    python icae_raw_labels.py
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

CANDIDATES = [DATA / "icae_tree_field23.json", DATA / "icae_tree.json"]
# last commit carrying icae_tree.json (pruned in fbf0c11); older, looser prompt
TREE_COMMIT = "b79ea2a"
TREE_PATH = "data/labels/icae_tree.json"


def load_tree():
    for c in CANDIDATES:
        if c.exists():
            return json.loads(c.read_text()), str(c)
    print("WARNING: falling back to the archived tree, which used the older "
          "two-to-five-word LABEL_PROMPT. Its labels are NOT budget-matched to "
          "Doc2LoRA. Re-run icae_label.py to produce icae_tree_field23.json.")
    blob = subprocess.check_output(
        ["git", "show", f"{TREE_COMMIT}:{TREE_PATH}"],
        cwd=HERE.parents[1],
    )
    return json.loads(blob), f"{TREE_COMMIT}:{TREE_PATH} (LEGACY PROMPT)"


def main():
    tree, src = load_tree()
    out = {}

    def walk(n):
        kind = n.get("kind")
        if kind in ("field", "div", "sub") and "gen" in n:
            code = str(n["fid"]) if kind == "field" else str(n["ref"])
            out[code] = n["gen"]
        for c in n.get("children", []):
            walk(c)

    walk(tree)
    (DATA / "icae_raw.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote icae_raw.json ({len(out)} nodes) from {src}")


if __name__ == "__main__":
    main()
