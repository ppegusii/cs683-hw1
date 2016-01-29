PDF = report.pdf
TEX = report.tex

pdf:	$(TEX) 
	pdflatex report
	# pdflatex report
	# bibtex report
	# pdflatex report
	# pdflatex report

draft:	$(PDF)
	mv report.pdf draft.pdf

clean:
	rm -f $(PS) $(PDF) *.dvi *.log *.aux *.bbl *.blg
