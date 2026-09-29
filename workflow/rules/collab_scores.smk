# Build reusable collaboration per-pair score files (easy + hard) per field, in parallel.
# Metrics (AUC/AP/hits@k) are then computed instantly from the parquets by collab_metrics.py.
from os.path import join as j

COLLAB_WINDOWS = {
    "aps": "2000,2004,2008",
    "economics": "2008,2012,2016",
    "psychology": "2008,2012,2016",
    "chemistry": "2008,2012,2016",
    "arxiv_math": "2010,2014,2018",
    "arxiv_cs": "2010,2014,2018",
}

wildcard_constraints:
    field="|".join(COLLAB_WINDOWS),

rule save_collab_scores:
    output:
        hard=j(DATA_DIR, "collab_scores_{field}_hard.parquet"),
        easy=j(DATA_DIR, "collab_scores_{field}_easy.parquet"),
    params:
        field=lambda w: w.field,
        windows=lambda w: COLLAB_WINDOWS[w.field],
        dy=3,
    resources:
        mem_gb=25,
    script:
        "../scripts/save_collab_scores.py"

rule collab_scores_all:
    input:
        expand(j(DATA_DIR, "collab_scores_{field}_hard.parquet"), field=list(COLLAB_WINDOWS)),
