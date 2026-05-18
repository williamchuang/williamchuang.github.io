#!/bin/bash
# Run this from the root of your repo:
# cd ~/path/to/williamchuang.github.io
# bash fix_scripts.sh

set -e
REPO="."

# Files to fix (live pages only — skip index_ drafts)
FILES=(
  "Research/index.html"
  "GRAIL/index.html"
  "CEAS/index.html"
  "Ψ-Operator-Framework/index.html"
  "MSIA/index.html"
  "λ‑Stack Transformers/index.html"
  "_layouts/default.html"
)

POLYFILL='<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>'
MATHJAX_PATTERN='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

for rel in "${FILES[@]}"; do
  f="$REPO/$rel"
  if [ ! -f "$f" ]; then
    echo "SKIP (not found): $rel"
    continue
  fi

  # 1. Remove polyfill.io line entirely
  sed -i '' "/<script[^>]*polyfill\.io[^>]*><\/script>/d" "$f"

  # 2. Count remaining MathJax script tags
  count=$(grep -c "$MATHJAX_PATTERN" "$f" 2>/dev/null || true)

  if [ "$count" -gt 1 ]; then
    # Keep only the FIRST occurrence, remove all subsequent ones
    # Strategy: remove every MathJax script tag, then re-insert one clean one
    # after the first <head> or before </head>
    python3 - "$f" << 'PYEOF'
import sys, re

path = sys.argv[1]
with open(path) as fh:
    page = fh.read()

# Remove ALL MathJax script tags (any variant: async, defer, id=MathJax-script)
page = re.sub(
    r'<script[^>]*cdn\.jsdelivr\.net/npm/mathjax@3/es5/tex-mml-chtml\.js[^>]*>\s*</script>\n?',
    '', page
)
# Also remove the dynamic-loader pattern (s.src = '...mathjax...')
page = re.sub(
    r'<script>[^<]*s\.src\s*=\s*[\'"]https://cdn\.jsdelivr\.net/npm/mathjax@3[^<]*</script>\n?',
    '', page
)

# Re-insert exactly one clean MathJax tag before </head>
clean_tag = '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>\n'
if '</head>' in page and clean_tag not in page:
    page = page.replace('</head>', clean_tag + '</head>', 1)
elif clean_tag not in page:
    # No </head> — insert after opening <head> tag
    page = re.sub(r'(<head[^>]*>)', r'\1\n' + clean_tag, page, count=1)

with open(path, 'w') as fh:
    fh.write(page)
print(f"  Fixed duplicates in: {path}")
PYEOF
  fi

  remaining_polyfill=$(grep -c "polyfill.io" "$f" 2>/dev/null || true)
  remaining_mathjax=$(grep -c "$MATHJAX_PATTERN" "$f" 2>/dev/null || true)
  echo "OK: $rel  [polyfill=$remaining_polyfill, mathjax=$remaining_mathjax]"
done

echo ""
echo "Done. Review the counts above — polyfill should be 0, mathjax should be 1 for each."
echo ""
echo "Now commit:"
echo "  git add -A"
echo "  git commit -m 'Remove polyfill.io and deduplicate MathJax across all pages'"
echo "  git push"
