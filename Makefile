# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

pages: html
	cd build/html && tar zcf /tmp/comparts431-html.tar.gz . 
	tar zcf /tmp/comparts431-backup-`date "+%Y%m%d%H%M%S"`.tar.gz . 
	rm -rf build
	git checkout gh-pages 
	git reset --hard 
	git clean -f -d 
	tar zxf /tmp/comparts431-html.tar.gz 
	git add * 
	git add -u * 
	git commit -m "Updated gh-pages"
	git push
	git checkout main

backup:
	tar zcvf /tmp/comparts431.tar.gz source

restore:
	tar zxvf /tmp/comparts431.tar.gz
