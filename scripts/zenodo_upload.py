#!/usr/bin/env python3
"""Upload the artifact bundles to Zenodo as a DRAFT deposition.

The script creates the deposition, attaches the tarballs, and writes the
metadata. It deliberately stops there: publishing is irreversible (a published
record cannot be deleted and its files cannot be changed), so the last step is
yours, either in the web UI or with `--publish` once you have checked the draft.

Setup
-----
Make a personal access token at https://zenodo.org/account/settings/applications/
with the `deposit:write` and `deposit:actions` scopes, then:

    export ZENODO_TOKEN=...            # or put ZENODO_TOKEN=... in .env

Usage
-----
    python scripts/zenodo_upload.py --dry-run                       # show the plan
    python scripts/zenodo_upload.py dist/doc2lora-results.tar.zst   # draft + upload
    python scripts/zenodo_upload.py dist/*.tar.zst
    python scripts/zenodo_upload.py --deposition 1234567 dist/doc2lora-s2and.tar.zst
    python scripts/zenodo_upload.py --deposition 1234567 --publish  # irreversible

Use `--sandbox` to rehearse the whole thing against sandbox.zenodo.org (separate
account, separate token, nothing permanent).

After publishing, put the record id in scripts/fetch_artifacts.py (ZENODO_RECORD)
so `python scripts/fetch_artifacts.py results` works with no arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = "https://zenodo.org/api"
SANDBOX = "https://sandbox.zenodo.org/api"

TITLE = ("Doc2LoRA idea genes: reproduction code and archived intermediates for "
         '"Doc2LoRA Provides Decodable Representations of Scientific Ideas"')

DESCRIPTION = """<p>The reproduction workflow and the archived intermediate results for the
paper. Everything needed to rebuild the manuscript's tables is here: the code is a
snapshot of the trimmed reproduction repository, and the bundles hold the computed
intermediates, so a reader does not have to re-derive them. Reproducing from the raw
corpora instead needs roughly 200 GB of intermediates and several GPU-days.</p>

<ul>
<li><b>doc2lora-embedding-code-&lt;sha&gt;.tar.gz</b> &mdash; the workflow itself: a Snakemake
pipeline trimmed to the dependency closure of what the manuscript reads (24 rule files;
<code>snakemake -n paper_assets</code> resolves 330 jobs from the raw corpora). Includes
<code>REPRODUCE.md</code>, which maps every figure, table and quoted number to the rule that
produces it, and <code>data/ARTIFACTS.tsv</code>, a SHA-256 per archived file.</li>
<li><b>doc2lora-results.tar.zst</b> (1227 files) &mdash; everything the manuscript's
tables and figures are computed from that needs a GPU, the licensed APS text, or an LLM
judge to produce: the paired per-unit score pools behind the bootstrap confidence
intervals, the invertible adapter weights, every method's raw cluster labels and the
judge verdicts, the pair-axis decodes (Doc2LoRA and ICAE), the PACS node set, and the
manuscript's own tables for comparison. With this bundle alone,
<code>snakemake paper_assets --rerun-triggers mtime</code> rebuilds every table and
figure the paper reads, running only CPU rules.</li>
<li><b>doc2lora-s2and.tar.zst</b> (50 files) &mdash; the gene and text embeddings for
the five author-name-disambiguation benchmarks (zbMATH, QIAN, ArnetMiner, PubMed,
KISTI), so the disambiguation rows can be re-scored from the vectors up.</li>
</ul>

<p>Unpack the code snapshot, then fetch and verify the bundles against
<code>data/ARTIFACTS.tsv</code> (SHA-256 per file):</p>

<pre>tar xzf doc2lora-embedding-code-&lt;sha&gt;.tar.gz &amp;&amp; cd doc2lora-embedding-&lt;sha&gt;
cp workflow/config.template.yaml workflow/config.yaml   # edit the paths
python scripts/fetch_artifacts.py results --record &lt;this record&gt;
snakemake paper_assets -j4 --rerun-triggers mtime      # rebuilds every table, CPU only</pre>

<p>The 644k-paper APS gene matrices (~137 GB) are not included: they exceed a
Zenodo record and are a deterministic function of the APS corpus plus the
published hypernetwork checkpoints (<code>snakemake all_embeddings</code>). The APS
corpus itself is licensed and is not redistributed here.</p>
"""

METADATA = {
    "metadata": {
        "upload_type": "dataset",
        "title": TITLE,
        "description": DESCRIPTION,
        "creators": [
            # Mansuri and Zachariah contributed equally.
            {"name": "Mansuri, Chand Sahil", "affiliation": "Binghamton University"},
            {"name": "Zachariah, Joel", "affiliation": "Binghamton University"},
            {"name": "Kojaku, Sadamori", "affiliation": "Binghamton University"},
        ],
        "license": "cc-by-4.0",
        "access_right": "open",
        "keywords": ["document embeddings", "LoRA", "hypernetwork", "science of science",
                     "reproducibility", "author name disambiguation", "Snakemake workflow"],
        "related_identifiers": [
            {"identifier": "https://github.com/skojaku/doc2lora-embedding",
             "relation": "isSupplementTo", "resource_type": "software"},
        ],
    }
}


def token() -> str:
    tok = os.environ.get("ZENODO_TOKEN", "").strip()
    if not tok:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ZENODO_TOKEN"):
                    tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        sys.exit("no ZENODO_TOKEN (export it, or add ZENODO_TOKEN=... to .env)")
    return tok


def api(base: str, method: str, path: str, tok: str, payload: dict | None = None) -> dict:
    url = f"{base}{path}{'&' if '?' in path else '?'}access_token={tok}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:                    # noqa: S310
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo {method} {path} -> {e.code}: {e.read().decode()[:500]}")


def put_file(bucket: str, path: Path, tok: str) -> None:
    """Files go to the bucket API, which streams; curl handles multi-GB uploads."""
    url = f"{bucket}/{path.name}?access_token={tok}"
    if not subprocess.run(["curl", "--version"], capture_output=True).returncode == 0:
        sys.exit("curl is required to stream multi-GB uploads")
    print(f"  uploading {path.name} ({path.stat().st_size / 1e9:.2f} GB)")
    r = subprocess.run(["curl", "-fsS", "--retry", "3", "-T", str(path), "-X", "PUT", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"upload failed for {path.name}: {r.stderr[:500]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="tarballs from scripts/make_artifact_bundle.py")
    ap.add_argument("--deposition", help="add to an existing draft instead of creating one")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--publish", action="store_true", help="publish the draft (IRREVERSIBLE)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    base = SANDBOX if args.sandbox else LIVE
    for f in args.files:
        if not f.exists():
            sys.exit(f"missing {f}")

    if args.dry_run:
        print(f"host      : {base}")
        print(f"deposition: {args.deposition or '(create new draft)'}")
        print(f"title     : {TITLE}")
        print(f"license   : {METADATA['metadata']['license']}")
        for f in args.files:
            print(f"upload    : {f}  ({f.stat().st_size / 1e9:.2f} GB)")
        print("publish   :", "YES (irreversible)" if args.publish else "no -- draft only")
        return 0

    tok = token()

    if args.deposition:
        dep = api(base, "GET", f"/deposit/depositions/{args.deposition}", tok)
    else:
        dep = api(base, "POST", "/deposit/depositions", tok, {})
        print(f"draft deposition {dep['id']} created")

    if args.files:
        bucket = dep["links"]["bucket"]
        for f in args.files:
            put_file(bucket, f, tok)

    dep = api(base, "PUT", f"/deposit/depositions/{dep['id']}", tok, METADATA)
    print(f"metadata written; draft: {dep['links'].get('html')}")

    if args.publish:
        confirm = os.environ.get("ZENODO_CONFIRM_PUBLISH", "")
        if confirm != "yes":
            sys.exit("refusing to publish without ZENODO_CONFIRM_PUBLISH=yes "
                     "(publishing a Zenodo record cannot be undone)")
        pub = api(base, "POST", f"/deposit/depositions/{dep['id']}/actions/publish", tok)
        print(f"PUBLISHED: {pub['links'].get('record_html')}  doi={pub.get('doi')}")
        print(f"now set ZENODO_RECORD = \"{pub['id']}\" in scripts/fetch_artifacts.py")
    else:
        print("draft only -- review it, then re-run with --publish (or publish in the web UI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
