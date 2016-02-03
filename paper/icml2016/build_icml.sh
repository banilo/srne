#!/usr/bin/env bash

# rm -rf $(biber --cache)
rm *.aux
rm *.bcf
rm *.bbl
rm *.blg
rm *.ist
rm *.alg
rm *.acr
rm *.acn
rm *.glsdefs
rm *.out
rm *.lot
rm *.log
rm *.lof
rm *.toc
rm *.run.xml
rm *.synctex.gz
pdflatex -interaction=nonstopmode -file-line-error-style -synctex=1 example_paper.tex

bibtex example_paper
pdflatex -interaction=nonstopmode -file-line-error-style -synctex=1 example_paper
pdflatex -interaction=nonstopmode -file-line-error-style -synctex=1 example_paper
# makeindex -s paper.idx -t paper.glsdef -o paper.acr paper.acn
makeglossaries example_paper
pdflatex -interaction=nonstopmodxe -file-line-error-style -synctex=1 example_paper

open example_paper.pdf