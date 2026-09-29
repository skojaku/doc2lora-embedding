# ICAE memory-slot embeddings, and the citation transform trained on them.
#
# ICAE compresses a document into 128 memory slots of 4,096 channels and a frozen
# decoder reads them back. In this paper it is a DECODING comparison — it is not a
# row of the similarity table — so what is reproduced here is only what a reported
# result reads:
#
#   App. symmetric-adapter gives every space the same citation transform and finds
#   that only the compressed generator spaces gain (ICAE +.178 over 13 of 14
#   benchmarks, ahead of \doctolora's +.059). Those numbers read `icae_emb.npz` and
#   `icae_genkron_emb.npz` through groupc_bench.smk:GCB_CACHED and
#   gcb_s2and_pool.py.
#
# The retrieval-evaluation half that used to live here (ic_eval_field/aps/s2and,
# writing results_*_icae.csv) is deliberately absent: no table, figure or sentence
# in the paper reads those files.
#
# The transform is trained here rather than by gcb_train because the ICAE space
# factorises over 128 memory tokens x 4,096 channels, which gcb_train's flat path
# does not cover.
#
# The paper's ICAE artifacts were produced once and archived; `fetch_artifacts.py
# results` is the cheap way to obtain them. These rules are the expensive way, and
# the only way to obtain them from the corpus.
#
# RUN: snakemake icae_embeddings -j2
import os
from os.path import join as j

IC = j(DATA_DIR, "icae")
S2 = j(DATA_DIR, "s2and")
IC_FIELDS = config.get("icae_fields", ["economics", "psychology"])
IC_S2AND = config.get("icae_s2and", ["zbmath", "qian", "arnetminer", "pubmed", "kisti"])
IC_ENV = ("set -a; source .env 2>/dev/null || true; set +a; "
          f"export HF_HOME={os.path.abspath(config.get('hf_home', 'data/agent_assets/hf_cache'))} "
          "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;")
IC_GPU = f"bash {SCRIPTS}/gpu_run.sh"
IC_OAX_EMB = j(DATA_DIR, "fields", "{field}", "embeddings")
IC_APS_EMB = j(DATA_DIR, "aps", "embeddings")

# The eval-id subsets come from bench.smk:bench_ids, which computes the same union
# for the decontamination chain. One producer, not two.
IC_IDS = j(IC, "{field}_eval_ids.parquet")

wildcard_constraints:
    field="|".join(IC_FIELDS + ["aps"]),
    ds="|".join(IC_S2AND),


# ── compress the OpenAlex training pool into per-token slots ──────────────
rule ic_pool_slots:
    input:
        pool=GCB_POOL_TEXT,
    output:
        slots=j(IC, "pool_slots.npy"),
        pids=j(IC, "pool_pids.npy"),
    resources:
        gpu=1, mem_gb=40,
    shell:
        IC_ENV + " BATCH=24 " + IC_GPU + f" python {SCRIPTS}/embed_pool_slots.py"


# Trained on the SAME citation sample as the reported g_theta (see the GOTCHA in
# groupc_bench.smk): the arms of the symmetric control must differ only in the
# space they act on, never in the supervision they were given.
rule ic_train:
    input:
        slots=j(IC, "pool_slots.npy"),
        trip=GCB_TRIPLETS,
    output:
        adapter=j(IC, "adapter_icae.pt"),
    resources:
        gpu=1, mem_gb=40,
    shell:
        IC_ENV + " STEPS=8000 " + IC_GPU + f" python {SCRIPTS}/train_icae_adapter.py"


# ── embed each benchmark subset: raw slots, and slots through the transform ──
rule ic_embed_field:
    input:
        adapter=j(IC, "adapter_icae.pt"),
        ids=j(IC, "{field}_eval_ids.parquet"),
    output:
        raw=j(IC_OAX_EMB, "icae_emb.npz"),
        gk=j(IC_OAX_EMB, "icae_genkron_emb.npz"),
    wildcard_constraints:
        field="|".join(IC_FIELDS),
    resources:
        gpu=1, mem_gb=40,
    shell:
        IC_ENV + " BATCH=24 " + IC_GPU + f" python {SCRIPTS}/embed_apply.py field {{wildcards.field}}"


rule ic_embed_aps:
    input:
        adapter=j(IC, "adapter_icae.pt"),
        ids=j(IC, "aps_eval_ids.parquet"),
    output:
        raw=j(IC_APS_EMB, "icae_emb.npz"),
        gk=j(IC_APS_EMB, "icae_genkron_emb.npz"),
    resources:
        gpu=1, mem_gb=40,
    shell:
        IC_ENV + " BATCH=24 " + IC_GPU + f" python {SCRIPTS}/embed_apply.py aps"


rule ic_embed_s2and:
    input:
        adapter=j(IC, "adapter_icae.pt"),
    output:
        raw=j(S2, "proc", "{ds}", "icae.npz"),
        gk=j(S2, "proc", "{ds}", "icae_genkron.npz"),
    resources:
        gpu=1, mem_gb=20,
    shell:
        IC_ENV + " BATCH=24 " + IC_GPU + f" python {SCRIPTS}/embed_apply.py s2and {{wildcards.ds}}"


rule icae_embeddings:
    input:
        expand(j(IC_OAX_EMB, "icae_genkron_emb.npz"), field=IC_FIELDS),
        j(IC_APS_EMB, "icae_genkron_emb.npz"),
        expand(j(S2, "proc", "{ds}", "icae_genkron.npz"), ds=IC_S2AND),
