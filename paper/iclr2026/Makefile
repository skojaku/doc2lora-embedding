# Build the paper. We use XeLaTeX (the preamble's fontspec branch needs it for
# real Times bold); latexmk runs xelatex + bibtex to convergence, so a plain
# `make` resolves all citations and cross-references in one invocation. The
# bibliography sits before the appendices, which add further citations, so a
# partial build can leave stale `?`/`????` markers; `make check` fails loudly
# when that happens, and `make repro` forces a clean, deterministic rebuild.
#
# TARGET ARTIFACT: an anonymous ICLR 2027 submission (plain
# \usepackage{iclr2027_conference} in main.tex, no [preprint] option) -- the
# style renders "Anonymous authors / Paper under double-blind review" instead
# of the \author block. To produce a non-anonymous arXiv preprint instead, pass
# [preprint] in main.tex; for camera-ready, use \iclrfinalcopy.

LATEXMK = latexmk -xelatex -interaction=nonstopmode -halt-on-error

all: main.pdf

main.pdf: main.tex main.bib $(wildcard sections/*.tex)
	$(LATEXMK) main.tex
	@$(MAKE) --no-print-directory check

# Guaranteed-deterministic build from a clean state.
repro:
	latexmk -C main.tex
	rm -f main.bbl main.run.xml
	$(LATEXMK) main.tex
	@$(MAKE) --no-print-directory check

# Fail if the last build left undefined citations or references behind.
check:
	@if grep -E "Reference .* undefined|Citation .* undefined|There were undefined references" main.log; then \
		echo "ERROR: unresolved references/citations remain (see main.log)"; exit 1; \
	fi
	@echo "check: no unresolved references or citations."

clean:
	latexmk -C main.tex
	rm -f main.bbl main.run.xml

.PHONY: all repro check clean
