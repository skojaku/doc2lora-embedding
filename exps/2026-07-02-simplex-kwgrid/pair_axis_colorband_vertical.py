"""Narrow, rotated (vertical-band) colorband figure, both methods annotated.

Two vertical bands per example (Doc2LoRA, In-context), 0% at top / 100% at bottom.
Both bands are labeled with hand-written summaries at 25/50/75%, each pointing via
a leader line + circle marker at the exact cell it describes (D2L text to the left
of its band, In-context text to the right of its band, mirrored). No paper
titles/DOIs in the figure; name them in surrounding prose instead.

In-context's summaries are frequently near-identical across all three quarters --
that repetition is the evidence that it produces a fixed hybrid framing regardless
of the requested mixing weight, and is left to speak for itself (no editorializing
in the caption text; see feedback_figure_annotations memory).

Reuses pair_axis_colorband.py's corner-loading/SBERT-projection/blend-color
internals against Mistral-backbone (tag=_pairaxisM) decode results.

  PYTHONPATH=$DOC_TO_LORA_SRC \
    python pair_axis_colorband_vertical.py
"""
import os, sys, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_common as sc
import pair_axis_colorband as pc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

CB = sns.color_palette('colorblind')
BLUE, ORANGE = np.array(CB[0]), np.array(CB[1])

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

R = sc.GRID_DEN
n = R + 1
HERE = os.path.dirname(os.path.abspath(__file__))
tag = '_pairaxisM'

from sentence_transformers import SentenceTransformer
sb = SentenceTransformer('all-mpnet-base-v2', device='cuda')
def emb(t):
    return unit(np.atleast_2d(sb.encode(t, normalize_embeddings=True, show_progress_bar=False)).ravel())

D2L_TEXT = {
    'pairL5_17': {
        '0.25': 'Two coupled subnetworks -- social agents and political opinion -- are linked through a physics-style operator that lets belief and interaction co-evolve.',
        '0.5': 'Opinion dynamics are recast with a spatial nonlinearity and competition term borrowed directly from nonlinear-optics formalism.',
        '0.75': 'Harmonic generation and enhanced nonlinear optical response emerge from a quasimonochromatic component interacting with a birefringent one.',
    },
    'pairL5_06': {
        '0.25': "DFT and first-principles calculations find UFe4Al7's magnetic structure is noncollinear, contradicting symmetry-analysis predictions.",
        '0.5': 'Gene duplication drives shifts in magnetic structure in a modeled bacterium, combining DFT-style calculations with Monte Carlo genome simulations.',
        '0.75': 'Gene duplication and transposon mobility trade off in microbial genomes after a sudden environmental shift, shaped by selection.',
    },
    'pairL5_27': {
        '0.25': 'A fast, scalable algorithm finds densely-connected community structure in large graphs, far faster than prior community-detection methods.',
        '0.5': 'The same fast community-detection framing is applied to many-body nuclear physics, treating nucleon interactions as a network to partition.',
        '0.75': 'A unitary dynamical model analyzes meson-exchange interactions between nucleons via N-body partial-wave analysis.',
    },
    'pairL5_00': {
        '0.25': 'Picosecond-pulse laser crystallization and calorimetry probe how Pd77Si23 metallic glass nucleates, addressing gaps in understanding nucleation mechanisms.',
        '0.5': 'Microwave-pulse-driven graph spectral algorithms analyze community structure in protein-phenotype networks and nanostructured materials alike.',
        '0.75': "Spectral methods based on the graph Laplacian's eigenvectors identify densely-interconnected communities via modularity maximization and graph partitioning.",
    },
    'pairL5_07': {
        '0.25': 'Risk perception shapes epidemic spreading of a bacterial-infection model, studied via mean-field analysis across regular, random, and scale-free networks.',
        '0.5': 'Resonant photon irradiation alters the morphology and infectivity of a bacteriophage-infected nanofilm, linking photodynamic and thermal-desorption effects.',
        '0.75': 'Resonant infrared photothermal spectroscopy probes how methanol desorbs from a Cu(111) surface at 90K.',
    },
    'pairL5_30': {
        '0.25': 'A network built from a cataloged earthquake sequence reveals correlations between earthquakes and their aftershocks.',
        '0.5': 'Correlations and alignment structures among earthquakes in the same region are quantified with statistical network methods, illustrated for the 1981 Cascadia sequence.',
        '0.75': 'A cranked-shell-model calculation with multiple quadrupole moments analyzes collective alignment behavior in the 152Sm nucleus.',
    },
}

IC_TEXT = {
    'pairL5_17': {
        '0.25': 'A hybrid framework combining political-opinion-network evolution with the optical response of nonlinear composite materials.',
        '0.5': 'A hybrid framework combining political-opinion-network evolution with the optical response of nonlinear composite materials.',
        '0.75': 'A hybrid framework combining political-opinion-network evolution with the optical response of nonlinear composite materials.',
    },
    'pairL5_06': {
        '0.25': "A symmetry-driven magneto-genomic evolution model linking UFe5Al7's magnetic structure to microbial genomic adaptation.",
        '0.5': "A symmetry-driven magneto-genomic evolution model linking UFe5Al7's magnetic structure to microbial genomic adaptation under environmental change.",
        '0.75': "A symmetry-driven magneto-genomic evolution model linking UFe5Al7's noncollinear magnetic structure to microbial genomic evolution.",
    },
    'pairL5_27': {
        '0.25': 'A fast community-detection algorithm applied to collaboration and interaction networks in nuclear physics.',
        '0.5': 'A fast community-detection algorithm applied to collaboration and interaction networks in nuclear physics.',
        '0.75': 'A fast community-detection algorithm applied to collaboration and interaction networks in nuclear physics.',
    },
    'pairL5_00': {
        '0.25': 'Picosecond laser annealing of metallic glasses blended with spectral community detection in material-defect networks.',
        '0.5': 'The same blend, re-framed around community detection in glass-formation networks rather than material-defect networks.',
        '0.75': 'The same blend as the 25% cell, back to material-defect networks in the Pd77Si23 glassy-metal system.',
    },
    'pairL5_07': {
        '0.25': 'Risk-perception-mediated epidemic spreading combined with methanol adsorption on copper surfaces, weighted mostly toward epidemic modeling.',
        '0.5': 'The same epidemic/methanol-adsorption combination, now framed around resonant IR laser stimulation of the reactive surface.',
        '0.75': 'The same risk-perception/methanol-adsorption combination as the 25% cell, re-labeled to a different weighting.',
    },
    'pairL5_30': {
        '0.25': 'Earthquake correlations in seismic regions combined with nuclear excitations in the transitional 157Tm nucleus, weighted mostly toward earthquakes.',
        '0.5': 'The same earthquake/nuclear combination, framed around the shared N=88 transitional-nucleus and seismic-network methods.',
        '0.75': 'The same earthquake/nuclear combination as the 25% cell, now weighted mostly toward the nuclear side.',
    },
}

FIELD_LABELS = {
    'pairL5_17': 'Social science ↔ physics',
    'pairL5_06': 'Condensed matter ↔ biology',
    'pairL5_27': 'Network science ↔ nuclear physics',
    'pairL5_00': 'Materials science ↔ network science',
    'pairL5_07': 'Epidemiology ↔ surface chemistry',
    'pairL5_30': 'Geophysics ↔ nuclear physics',
}

FIG_W = 6.4
CELL_H = 0.30
BAND_H = CELL_H * n
BAND_W = 0.30
BAND_GAP = 0.10
D2L_X0 = 2.55
IC_X0 = D2L_X0 + BAND_W + BAND_GAP
TITLE_H = 0.24
TOP_PAD = 0.30
BOT_PAD = 0.14
CALLOUT_WRAP = 31

def make_figure(set_name, out_path):
    spec = sc.load_corners(sc.corners_path(set_name))
    A, B = spec['leads']['A'], spec['leads']['B']
    eA, eB = emb(A), emb(B)
    axis_vec = eB - eA
    denom = float(axis_vec @ axis_vec) + 1e-12
    fpath = os.path.join(sc.RESULTS, f'absfollow_{set_name}{tag}.json')
    cells = {tuple(c['bary']): c for c in json.load(open(fpath))['cells']}

    def band_values(method):
        vals = []
        for k in range(n):
            c = cells[(R - k, k, 0)]
            ab = c[method]['abs']
            d = emb(ab)
            t = float((d - eA) @ axis_vec / denom)
            vals.append(t)
        return vals

    wrapped_title = '\n'.join(textwrap.wrap(FIELD_LABELS[set_name], 32))
    title_lines = wrapped_title.count('\n') + 1
    block_h = TITLE_H * title_lines + TOP_PAD + BAND_H + BOT_PAD
    FIG_H = 0.06 + block_h + 0.06

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
    ax.axis('off')
    ax.invert_yaxis()

    top = 0.06
    ax.text(FIG_W / 2, top, wrapped_title, fontsize=14, fontweight='bold', ha='center', va='top', linespacing=1.15)
    band_top = top + TITLE_H * title_lines + TOP_PAD
    band_bottom = band_top + BAND_H
    band_cx = (D2L_X0 + IC_X0 + BAND_W) / 2

    for method, x0 in [('doc2lora', D2L_X0), ('incontext', IC_X0)]:
        vals = band_values(method)
        cols = [pc.blend(v) for v in vals]
        for k in range(n):
            y = band_top + k * CELL_H
            ax.add_patch(patches.Rectangle((x0, y), BAND_W, CELL_H,
                                           facecolor=cols[k], edgecolor='0.55', lw=0.4))
    ax.text(D2L_X0 + BAND_W / 2, band_top - 0.03, 'D2L', fontsize=9, ha='center', va='bottom', color='0.25')
    ax.text(IC_X0 + BAND_W / 2, band_top - 0.03, 'IC', fontsize=9, ha='center', va='bottom', color='0.25')

    ax.add_patch(patches.Rectangle((D2L_X0, band_top - 0.045), IC_X0 + BAND_W - D2L_X0, 0.035,
                                   facecolor=tuple(BLUE)))
    ax.add_patch(patches.Rectangle((D2L_X0, band_bottom + 0.013), IC_X0 + BAND_W - D2L_X0, 0.035,
                                   facecolor=tuple(ORANGE)))
    ax.text(band_cx, band_top - 0.19, '0% (A)', fontsize=8, ha='center', va='bottom', color=tuple(BLUE))
    ax.text(band_cx, band_bottom + 0.10, '100% (B)', fontsize=8, ha='center', va='top', color=tuple(ORANGE))

    for frac in [0.25, 0.5, 0.75]:
        k = round(frac * R)
        y = band_top + (k + 0.5) * CELL_H

        cx_d2l = D2L_X0 + BAND_W / 2
        summ_d2l = D2L_TEXT.get(set_name, {}).get(str(frac), '')
        wrapped_d2l = '\n'.join(textwrap.wrap(summ_d2l, CALLOUT_WRAP, break_long_words=False))
        ax.plot([cx_d2l, D2L_X0 - 0.05, 2.45], [y, y, y], color='0.6', lw=0.6)
        ax.plot(cx_d2l, y, 'o', ms=7, mfc='white', mec='black', mew=1.1, zorder=5)
        ax.text(2.43, y, wrapped_d2l, fontsize=8, color='0.12', ha='right', va='center', linespacing=1.3)

        cx_ic = IC_X0 + BAND_W / 2
        summ_ic = IC_TEXT.get(set_name, {}).get(str(frac), '')
        wrapped_ic = '\n'.join(textwrap.wrap(summ_ic, CALLOUT_WRAP, break_long_words=False))
        ax.plot([cx_ic, IC_X0 + BAND_W + 0.05, IC_X0 + BAND_W + 0.09], [y, y, y], color='0.6', lw=0.6)
        ax.plot(cx_ic, y, 'o', ms=7, mfc='white', mec='black', mew=1.1, zorder=5)
        ax.text(IC_X0 + BAND_W + 0.07, y, wrapped_ic, fontsize=8, color='0.12', ha='left', va='center', linespacing=1.3)

    fig.savefig(out_path, dpi=300, transparent=True)
    print('saved', out_path, FIG_W, FIG_H)

OUTDIR = os.path.join(HERE, 'figs', 'colorband-gallery')
os.makedirs(OUTDIR, exist_ok=True)
for s in ['pairL5_17', 'pairL5_06', 'pairL5_27', 'pairL5_00', 'pairL5_07', 'pairL5_30']:
    make_figure(s, os.path.join(OUTDIR, f'colorband-vertical-{s}.pdf'))
