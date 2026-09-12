#!/bin/bash
# Regenerate every computed value and confirm the PDF agrees with the artefacts.
# Run before every build that leaves this machine.
set -e
P=/ANON/miniconda3/envs/federatedlearning/bin/python
cd "$(dirname "$0")/paper"
echo "== regenerating computed values from run artefacts =="
$P figures/computed_values.py | sed 's/^/   /'
echo "== regenerating figures from run artefacts =="
$P figures/make_figures.py 2>&1 | tail -2 | sed 's/^/   /'
echo "== rebuilding =="
pdflatex -interaction=nonstopmode main.tex > /tmp/vb.log 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /tmp/vb.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex > /tmp/vb.log 2>&1; RC=$?
E=$(grep -cE "^! " /tmp/vb.log || true); echo "   latex errors: $E (pdflatex exit $RC)"
if [ "$E" != "0" ] || [ "$RC" != "0" ]; then
  echo "   FAIL: LaTeX did not build cleanly"; grep -E "^! |^l\.[0-9]" /tmp/vb.log | head -8; exit 1
fi
P2=$(pdftotext main.pdf - | grep -c '\*\*' || true)
if [ "$P2" != "0" ]; then echo "   FAIL: $P2 literal markdown ** in the PDF"; exit 1; fi
echo "== stale hand-entered corpus numbers =="
if grep -rnE "\b(79 runs|927|76 runs|802 invocations|\+0\.195|\+0\.514|\+0\.671|2743)\b" sections/*.tex main.tex figures/make_figures.py; then
  echo "   FAIL: hand-entered corpus statistics found - use the macros"; exit 1
else echo "   none"; fi
echo "== anonymity =="; cd .. && ./check_anonymity.sh | tail -1
