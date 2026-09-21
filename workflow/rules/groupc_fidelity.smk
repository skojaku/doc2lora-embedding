# Decode fidelity at the document points (#95).
#
# STATUS: NOT RUN BY DECISION.  Decode fidelity at single-document points is a property of the
# doc-to-LoRA hypernetwork itself, established in that paper; #95 is answered by citing the numbers
# reported there rather than re-measuring them here.  This chain is kept for two reasons: `fid_sample`
# is the APS paper sample that #73 (prompt sensitivity) and #71 (throughput) draw on, and the
# measurement is one command away (`snakemake groupc_fidelity`) if it is ever wanted.
#
# The manuscript never measures whether a decoded adapter is faithful to the paper it came from: the
# evidence for Contribution 1 is one cacio e pepe recipe + one kake udon counterpart, in App. D, on
# Mistral-7B, while every quantitative result uses Qwen3-4B.  This chain measures it on real papers
# with the reported encoder, and reports a ceiling and two floors on the same units:
#
#   fid_sample -> sample.parquet            N uniform APS papers with a real abstract
#   fid_decode -> decodes_{cond}.json       cond in {d2l, incontext, mismatch, prior}
#   fid_score  -> fid_scores.json + figs/decode_fidelity.{tex,pdf}
#                 + figs/decode_fidelity_examples.tex  (verbatim Qwen3-4B decodes for App. D)
#
# Instruments are deterministic (lexical overlap, keyword recall, SBERT cosine, and the rank of the
# true source among all N papers) -- no LLM judge, so #107's circularity does not apply here.
#
# RUN: snakemake groupc_fidelity -j4 --rerun-triggers mtime
from os.path import join as j

FIGS_DIR = config.get("figs_dir", "figs")
FID_DIR = j("data", "groupc", "fidelity")
FID_N = config.get("fid_n_papers", 200)
FID_SEED = config.get("fid_seed", 0)
FID_MIN_CHARS = config.get("fid_min_abstract_chars", 400)
FID_CONDS = config.get("fid_conditions", ["d2l", "incontext", "mismatch", "prior"])
FID_MAXNEW = config.get("fid_max_new_tokens", 160)
FID_EXAMPLES = config.get("fid_n_examples", 3)
FID_SBERT = config.get("compat_sbert_model", "sentence-transformers/all-mpnet-base-v2")

FID_SAMPLE = j(FID_DIR, "sample.parquet")
FID_SCORES = j(FID_DIR, "fid_scores.json")
FID_ROWS = j(FID_DIR, "fid_rows.parquet")
FID_TABLE = j(FIGS_DIR, "decode_fidelity.tex")
FID_EXTEX = j(FIGS_DIR, "decode_fidelity_examples.tex")
FID_FIG = j(FIGS_DIR, "decode_fidelity.pdf")

D2L_SRC = config.get("doc_to_lora_src", "doc-to-lora/src")
D2L_ENV = (
    "set -a; source .env 2>/dev/null; set +a; "
    f"export DOC_TO_LORA_SRC={D2L_SRC} PYTHONPATH={D2L_SRC} "
    f"HF_HOME={config.get('hf_home', 'data/agent_assets/hf_cache')} "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
)


rule fid_sample:
    input:
        paper_text=ancient(j(APS_DIR, "paper_text.parquet")),
    output:
        sample=FID_SAMPLE,
    params:
        n_papers=FID_N,
        seed=FID_SEED,
        min_chars=FID_MIN_CHARS,
    resources:
        mem_gb=16,
    script:
        "../scripts/groupc/fid_sample.py"


rule fid_decode:
    input:
        sample=FID_SAMPLE,
        ckpt=ancient(QWEN_CHECKPOINT_PATH),
    output:
        decodes=j(FID_DIR, "decodes_{cond}.json"),
    params:
        ckpt=QWEN_CHECKPOINT_PATH,
        maxnew=FID_MAXNEW,
        seed=FID_SEED,
    wildcard_constraints:
        cond="d2l|incontext|mismatch|prior",
    resources:
        gpu=1,
        mem_gb=20,
    shell:
        D2L_ENV +
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/fid_decode.py --cond {wildcards.cond} "
        "--sample {input.sample} --out {output.decodes} --ckpt {params.ckpt} "
        "--max_new_tokens {params.maxnew} --seed {params.seed}"


rule fid_score:
    input:
        sample=FID_SAMPLE,
        decodes=expand(j(FID_DIR, "decodes_{cond}.json"), cond=FID_CONDS),
    output:
        scores=FID_SCORES,
        rows=FID_ROWS,
        table=FID_TABLE,
        examples=FID_EXTEX,
        fig=FID_FIG,
    params:
        sbert_model=FID_SBERT,
        n_examples=FID_EXAMPLES,
    resources:
        gpu=1,
        mem_gb=20,
    script:
        "../scripts/groupc/fid_score.py"


rule groupc_fidelity:
    input:
        FID_SCORES,
        FID_TABLE,
        FID_EXTEX,
        FID_FIG,
