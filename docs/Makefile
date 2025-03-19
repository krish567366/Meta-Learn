# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Krishna BajpaiMetaLearn
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

quantum-view:
	@python -m http.server 8000 --directory "$(BUILDDIR)/html"