#!/usr/bin/env bash
#
# Assemble a ready-to-push Hugging Face Space directory from the project code
# plus the Space-specific files here — without committing duplicated code to the
# main repo. Produces deploy/hf_space/build/.
#
#   bash deploy/hf_space/build_space.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/build"

rm -rf "$OUT"
mkdir -p "$OUT/data"

# Space-specific files (entry point, deps, Space card)
cp "$HERE/app.py"          "$OUT/"
cp "$HERE/requirements.txt" "$OUT/"
cp "$HERE/README.md"       "$OUT/"

# Application code the chatbot needs at runtime
cp "$ROOT/data_pipeline.py" "$OUT/"
cp "$ROOT/deberta_model.py" "$OUT/"
cp "$ROOT/config.py"        "$OUT/"
cp "$ROOT/gradio_app.py"    "$OUT/"
cp -r "$ROOT/utils"         "$OUT/"
cp -r "$ROOT/agent"         "$OUT/"
cp "$ROOT/data/context_priors.json" "$OUT/data/"

# Never ship pycache
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "✓ Space assembled at: $OUT"
echo "  Files:"
ls "$OUT"
echo ""
echo "  Next: push \$OUT to your Space (see DEPLOY.md)."
