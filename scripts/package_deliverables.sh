#!/bin/bash
# Package OC deliverables into zip files
set -e

PROJECT="Demeule_Laureenda"
DATE="052026"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/livrables"
mkdir -p "$OUT"

echo "Packaging deliverables to $OUT"

# 1 - API (README + deployment info)
zip -r "$OUT/${PROJECT}_1_API_${DATE}.zip" \
  "$ROOT/README.md" \
  "$ROOT/deployment/" \
  "$ROOT/src/api/" \
  -x "*.pyc" "__pycache__/*"

# 2 - Scripts + notebooks
zip -r "$OUT/${PROJECT}_2_scripts_notebook_modelisation_${DATE}.zip" \
  "$ROOT/src/training/" \
  "$ROOT/notebooks/" \
  "$ROOT/mlflow/" \
  -x "*.pyc" "__pycache__/*" "*.ipynb_checkpoints/*"

# 3 - Full code repo
zip -r "$OUT/${PROJECT}_3_dossier_code_${DATE}.zip" \
  "$ROOT/src/" \
  "$ROOT/tests/" \
  "$ROOT/requirements.txt" \
  "$ROOT/README.md" \
  "$ROOT/.github/" \
  -x "*.pyc" "__pycache__/*" "*/__pycache__/*"

# 4 - Streamlit interface
zip -r "$OUT/${PROJECT}_4_interface_test_API_${DATE}.zip" \
  "$ROOT/streamlit_app/" \
  "$ROOT/.env.example" \
  -x "*.pyc"

# 5 - Blog
zip -r "$OUT/${PROJECT}_5_blog_${DATE}.zip" \
  "$ROOT/blog/"

# 6 - Presentation
zip -r "$OUT/${PROJECT}_6_presentation_${DATE}.zip" \
  "$ROOT/docs/"

echo "Done. Files in $OUT:"
ls -la "$OUT"

# Global OC folder zip
GLOBAL="$OUT/Realisez_une_analyse_de_sentiments_Demeule_Laureenda.zip"
zip -j "$GLOBAL" \
  "$OUT/${PROJECT}_1_API_${DATE}.zip" \
  "$OUT/${PROJECT}_2_scripts_notebook_modelisation_${DATE}.zip" \
  "$OUT/${PROJECT}_3_dossier_code_${DATE}.zip" \
  "$OUT/${PROJECT}_4_interface_test_API_${DATE}.zip" \
  "$OUT/${PROJECT}_5_blog_${DATE}.zip" \
  "$OUT/${PROJECT}_6_presentation_${DATE}.zip"
echo "Global zip: $GLOBAL"
