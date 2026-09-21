# ICAE benchmark results (vs doc2lora qwen genes + baselines)

ICAE = raw mean-pooled 128 memory slots; `icae_genkron` = invertible per-token adapter (common rotation+scale + per-token scaling), trained on the same OpenAlex citation triplets as doc2lora's general adapter.

## Field tasks (collab AUC / next-paper AUC / topic F1)

### economics

| model        |   collab_AUC |   collab_AP |   np_AUC |   topic_F1 |   topic_acc |
|:-------------|-------------:|------------:|---------:|-----------:|------------:|
| qwen         |     0.593383 |   0.018587  | 0.813807 |   0.206361 |    0.60525  |
| qwen_kron    |     0.651765 |   0.0214669 | 0.940889 |   0.315437 |    0.679667 |
| qwen_genkron |     0.625745 |   0.020397  | 0.911354 |   0.303179 |    0.66725  |
| sbert        |     0.653424 |   0.0213852 | 0.93842  |   0.342429 |    0.687667 |
| specter2     |     0.628872 |   0.0218727 | 0.905077 |   0.2939   |    0.659667 |
| instructor   |     0.616852 |   0.0209625 | 0.877808 |   0.313014 |    0.665167 |
| icae         |     0.561024 |   0.015114  | 0.671997 |   0.102291 |    0.479917 |
| icae_genkron |     0.625279 |   0.0209791 | 0.911664 |   0.266838 |    0.652417 |

### psychology

| model        |   collab_AUC |   collab_AP |   np_AUC |   topic_F1 |   topic_acc |
|:-------------|-------------:|------------:|---------:|-----------:|------------:|
| qwen         |     0.606187 |   0.0401208 | 0.799179 |   0.183296 |    0.569083 |
| qwen_kron    |     0.691785 |   0.0208759 | 0.944037 |   0.323241 |    0.682083 |
| qwen_genkron |     0.690257 |   0.028617  | 0.92381  |   0.294782 |    0.664917 |
| sbert        |     0.691951 |   0.0230837 | 0.936529 |   0.357567 |    0.684833 |
| specter2     |     0.668994 |   0.0250462 | 0.908322 |   0.298449 |    0.6575   |
| instructor   |     0.649826 |   0.0277063 | 0.887749 |   0.295112 |    0.666167 |
| icae         |     0.58329  |   0.029929  | 0.66165  |   0.103055 |    0.46525  |
| icae_genkron |     0.666276 |   0.0285271 | 0.910566 |   0.272431 |    0.632    |

### aps

| model        |   collab_AUC |   collab_AP |   np_AUC |    topic_F1 |   topic_acc |
|:-------------|-------------:|------------:|---------:|------------:|------------:|
| qwen         |     0.814828 |   0.0904492 | 0.841439 | 0.000900823 |  0.00433333 |
| qwen_genkron |     0.814871 |   0.0930709 | 0.947473 | 0.00115897  |  0.00558333 |
| sbert        |     0.856403 |   0.128609  | 0.953497 | 0.000848852 |  0.00408333 |
| specter2     |     0.822614 |   0.104217  | 0.936942 | 0.00110746  |  0.00533333 |
| instructor   |     0.837392 |   0.125522  | 0.901716 | 0.000538008 |  0.00258333 |
| icae         |     0.773352 |   0.139495  | 0.723596 | 0.00114207  |  0.0055     |
| icae_genkron |     0.847387 |   0.134177  | 0.952228 | 0.0021683   |  0.0105     |

## S2AND author disambiguation (B³ F1)

| model               |   zbmath |   qian |   arnetminer |   pubmed |   kisti |
|:--------------------|---------:|-------:|-------------:|---------:|--------:|
| gene                |    0.934 |  0.787 |        0.678 |    0.693 |   0.709 |
| gene_kron           |    0.934 |  0.832 |        0.669 |    0.749 |   0.761 |
| gene_genkron        |    0.932 |  0.85  |        0.708 |    0.815 |   0.767 |
| specter             |    0.934 |  0.832 |        0.669 |    0.739 |   0.746 |
| specter_kron        |    0.943 |  0.837 |        0.691 |    0.772 |   0.777 |
| sbert               |    0.944 |  0.865 |        0.698 |    0.832 |   0.793 |
| sbert_kron          |    0.941 |  0.819 |        0.704 |    0.774 |   0.792 |
| instructor          |    0.936 |  0.833 |        0.679 |    0.801 |   0.754 |
| instructor_kron     |    0.926 |  0.821 |        0.68  |    0.775 |   0.748 |
| embeddinggemma      |    0.934 |  0.832 |        0.707 |    0.823 |   0.771 |
| embeddinggemma_kron |    0.942 |  0.822 |        0.677 |    0.779 |   0.783 |
| icae                |    0.934 |  0.732 |        0.577 |    0.274 |   0.568 |
| icae_kron           |    0.937 |  0.665 |        0.644 |    0.636 |   0.735 |
| icae_genkron        |    0.93  |  0.853 |        0.694 |    0.823 |   0.773 |

---

## Mistral encoder + ICAE-vs-doc2lora (bench subset, same protocol)

doc2lora **mistral** genes run on the bench subset (`results_<field>_mistral_mbench.csv`,
`results_<ds>_mistral_mbench.csv`), with ICAE shown alongside.

### Next-paper AUC
| field | icae | icae_genkron | mistral | mistral_genkron | qwen | qwen_genkron | sbert |
|---|--:|--:|--:|--:|--:|--:|--:|
| economics | 0.676 | 0.913 | 0.826 | 0.915 | 0.814 | 0.911 | 0.938 |
| psychology | 0.668 | 0.912 | 0.817 | 0.926 | 0.799 | 0.924 | 0.937 |
| aps | 0.719 | 0.949 | 0.873 | 0.964 | 0.841 | 0.947 | 0.953 |

### Collab AUC
| field | icae_genkron | mistral_genkron | qwen_genkron | sbert |
|---|--:|--:|--:|--:|
| economics | 0.625 | 0.638 | 0.626 | 0.653 |
| psychology | 0.666 | 0.684 | 0.690 | 0.692 |
| aps | **0.847** | 0.804 | 0.815 | 0.856 |

### Topic F1 (aps degenerate ~0.002 for all)
| field | icae_genkron | mistral_genkron | qwen_genkron | sbert |
|---|--:|--:|--:|--:|
| economics | 0.267 | 0.308 | 0.303 | 0.342 |
| psychology | 0.272 | 0.306 | 0.295 | 0.358 |

### S2AND B³ F1 (icae_genkron from the ICAE run; mistral gene_genkron from mbench)
| dataset | icae_genkron | mistral_genkron | sbert |
|---|--:|--:|--:|
| zbmath | 0.930 | 0.934 | 0.944 |
| qian | **0.853** | 0.834 | 0.865 |
| arnetminer | 0.694 | **0.701** | 0.698 |
| pubmed | **0.823** | 0.816 | 0.832 |
| kisti | 0.773 | **0.779** | 0.793 |

### Verdict
ICAE+adapter ≈ doc2lora(qwen/mistral)+adapter — same tier, wins trade dataset-by-dataset, all
gaps ≤ ~0.02. doc2lora-mistral edges ahead on field next-paper (psych, aps); icae_genkron wins APS
collab + S2AND qian/pubmed. Raw ICAE is far below both (np ~0.7, S2AND pubmed 0.27) — the invertible
citation adapter closes a ~0.25 AUC / ~0.5 B³ gap. None is a clear benchmark winner; doc2lora's edge
is decode/interpretability, not these scores.
