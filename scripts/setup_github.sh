#!/bin/bash
# Initialize GitHub remote (run after creating repo on GitHub)
set -e
REPO_URL="${1:-}"
if [ -z "$REPO_URL" ]; then
  echo "Usage: ./scripts/setup_github.sh https://github.com/USER/air-paradis-sentiment.git"
  exit 1
fi
cd "$(dirname "$0")/.."
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git branch -M main
echo "Remote added. Run: git push -u origin main"
echo "Then add GitHub secrets: AZURE_CREDENTIALS, AZURE_WEBAPP_NAME"
