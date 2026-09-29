# Invertible Kronecker citation-adapter on frozen doc2lora genes (data/kron).
#
# Per field {field} with a citation network, for each gene encoder {enc} (gemma/qwen/mistral) and each
# text baseline (sbert/specter2/instructor):
#   kron_gene_adapter  : train+apply the gene adapter   -> {enc}_kron_emb.npz  (+ adapter_<field>_<enc>.pt)
#   kron_eval_gene     : 3-task eval {enc, enc_kron, text...} -> results_<field>_<enc>.csv
#
# Scripts are argparse CLIs (resolve their own emb_dir via bench_data.load); rules call them via shell
# and declare I/O for DAG tracking. GPU: shell inherits $CUDA_VISIBLE_DEVICES (export it to pin a device);
# resources `gpu=1` lets `--resources gpu=N` cap concurrency.
#
# RUN:  snakemake kron --rerun-triggers mtime -j1
#   (--rerun-triggers mtime is important: the gene/text npz already exist; without it Snakemake's
#    provenance check would try to RE-EXTRACT embeddings via the upstream embed_* rules.)
#   To pin a GPU:  CUDA_VISIBLE_DEVICES=1 snakemake kron --rerun-triggers mtime -j1
# Fields need a citation network (KRON_CITNET); arxiv_* have none and are excluded.
from os.path import join as j

KRON_DIR = "data/kron"
KRON_PREP = config.get("fields_prep_base", "/data/projects/gravity-of-ideas/tmp-data/preprocessed")
KRON_STEPS = config.get("kron_steps", 8000)

# fields that HAVE a citation network (required to train the adapter); arxiv_* have none -> excluded
KRON_CITNET = {
    "aps": config.get("aps_citation_net", "/data/datasets/aps/preprocessed/citation_net.npz"),
    "economics": j(KRON_PREP, "openalex-economics", "citation_net.npz"),
    "psychology": j(KRON_PREP, "openalex-psychology", "citation_net.npz"),
    "chemistry": j(KRON_PREP, "openalex-chemistry", "citation_net.npz"),
}

# gene encoders available per field (mistral genes only exist for APS)
KRON_GENE_ENC = config.get("kron_gene_enc", {
    "aps": ["gemma", "qwen", "mistral"],
    "economics": ["gemma", "qwen", "mistral"],
    "psychology": ["gemma", "qwen", "mistral"],
})
# OpenAlex-style fields live under data/fields/{field}/embeddings; APS under data/aps/embeddings
KRON_OAX_FIELDS = [f for f in KRON_GENE_ENC if f != "aps"]

OAX_EMB = j(DATA_DIR, "fields", "{field}", "embeddings")
APS_EMB = j(DATA_DIR, "aps", "embeddings")


# ── OpenAlex fields (economics / psychology / ...) ───────────────────────
rule kron_gene_adapter:
    input:
        gene=j(OAX_EMB, "{enc}_norm_lora_emb.npz"),
        cit=lambda w: KRON_CITNET[w.field],
    output:
        kron=j(OAX_EMB, "{enc}_kron_emb.npz"),
        adapter=j(KRON_DIR, "adapter_{field}_{enc}.pt"),
    params:
        field=lambda w: w.field, enc=lambda w: w.enc, steps=KRON_STEPS,
    resources:
        mem_gb=60, gpu=1,
    wildcard_constraints:
        field="|".join(KRON_OAX_FIELDS), enc="gemma|qwen|mistral",
    shell:
        "python {SCRIPTS}/train_apply.py --field {params.field} --enc {params.enc} --steps {params.steps}"

rule kron_eval_gene:
    input:
        base=j(OAX_EMB, "{enc}_norm_lora_emb.npz"),
        kron=j(OAX_EMB, "{enc}_kron_emb.npz"),
        sbert=j(OAX_EMB, "sbert_allmpnet.npz"),
        specter2=j(OAX_EMB, "baseline_specter2.npz"),
        instructor=j(OAX_EMB, "baseline_instructor.npz"),
        collab=j(DATA_DIR, "collab_scores_{field}_hard.parquet"),
        topics=j(DATA_DIR, "fields", "{field}", "paper_topics.parquet"),
    output:
        csv=j(KRON_DIR, "results_{field}_{enc}.csv"),
    params:
        field=lambda w: w.field, enc=lambda w: w.enc,
    resources:
        mem_gb=60, gpu=1,
    wildcard_constraints:
        field="|".join(KRON_OAX_FIELDS), enc="gemma|qwen|mistral",
    shell:
        "python {SCRIPTS}/eval_all.py --field {params.field} --enc {params.enc}"

# ── APS (different embedding dir + journal-code topics + mistral genes) ───
use rule kron_gene_adapter as kron_gene_adapter_aps with:
    input:
        gene=j(APS_EMB, "{enc}_norm_lora_emb.npz"),
        cit=KRON_CITNET["aps"],
    output:
        kron=j(APS_EMB, "{enc}_kron_emb.npz"),
        adapter=j(KRON_DIR, "adapter_aps_{enc}.pt"),
    params:
        field="aps", enc=lambda w: w.enc, steps=KRON_STEPS,
    wildcard_constraints:
        enc="|".join(KRON_GENE_ENC["aps"]),

use rule kron_eval_gene as kron_eval_gene_aps with:
    input:
        base=j(APS_EMB, "{enc}_norm_lora_emb.npz"),
        kron=j(APS_EMB, "{enc}_kron_emb.npz"),
        sbert=j(APS_EMB, "sbert_allmpnet.npz"),
        specter2=j(APS_EMB, "baseline_specter2.npz"),
        instructor=j(APS_EMB, "baseline_instructor.npz"),
        collab=j(DATA_DIR, "collab_scores_aps_hard.parquet"),
        topics=config.get("aps_paper_table", "/data/datasets/aps/preprocessed/paper_table.csv"),
    output:
        csv=j(KRON_DIR, "results_aps_{enc}.csv"),
    params:
        field="aps", enc=lambda w: w.enc,
    wildcard_constraints:
        enc="|".join(KRON_GENE_ENC["aps"]),

# ── aggregation ──────────────────────────────────────────────────────────
def _kron_targets():
    out = []
    for f, encs in KRON_GENE_ENC.items():
        for e in encs:
            out.append(j(KRON_DIR, f"results_{f}_{e}.csv"))
    return out

rule kron_all:
    input:
        _kron_targets(),
