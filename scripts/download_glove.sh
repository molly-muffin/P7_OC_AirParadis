#!/bin/bash
# Download GloVe embeddings (optional - pipeline uses gensim downloader by default)
set -e
DATA_DIR="$(dirname "$0")/../data/embeddings"
mkdir -p "$DATA_DIR"
echo "GloVe is loaded via gensim downloader (glove-twitter-25) during training."
echo "For manual download: https://nlp.stanford.edu/projects/glove/"
echo "Place glove.6B.100d.txt in $DATA_DIR if needed."
