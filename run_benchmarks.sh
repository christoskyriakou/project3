#!/bin/bash

# Configuration
DB="protein_vectors.fvecs"
QUERIES="targets.fasta"
SWISSPROT="swissprot_50k.fasta"
LOG_DIR="benchmarks"

mkdir -p $LOG_DIR

echo "=== Starting Protein Search Benchmarks ==="
echo "Note: Using w=3000 for LSH/Hypercube to handle large raw vector distances."

# --- TEST 1: FAST MODE ---
# k=4, L=3 means very few hash tables (fast but might miss neighbors)
echo "-----------------------------------------------------------------------"
echo "[TEST 1] Running FAST configuration..."
python3 protein_search.py \
    -d $DB -q $QUERIES -s $SWISSPROT \
    -o "$LOG_DIR/res_fast.txt" \
    -method all -N 50 \
    --lsh_k 4 --lsh_L 3 --lsh_w 3000 \
    --hc_k 3 --hc_probes 2 --hc_w 3000 \
    --ivf_k 50 --ivf_probe 1

# --- TEST 2: BALANCED MODE ---
# k=4, L=5 gives a good chance of collision without being too slow
echo "-----------------------------------------------------------------------"
echo "[TEST 2] Running BALANCED configuration..."
python3 protein_search.py \
    -d $DB -q $QUERIES -s $SWISSPROT \
    -o "$LOG_DIR/res_balanced.txt" \
    -method all -N 50 \
    --lsh_k 4 --lsh_L 5 --lsh_w 3000 \
    --hc_k 4 --hc_probes 10 --hc_w 3000 \
    --ivf_k 100 --ivf_probe 5

# --- TEST 3: ACCURATE MODE ---
# k=5, L=15 tries many hash tables to maximize recall
echo "-----------------------------------------------------------------------"
echo "[TEST 3] Running ACCURATE configuration..."
python3 protein_search.py \
    -d $DB -q $QUERIES -s $SWISSPROT \
    -o "$LOG_DIR/res_accurate.txt" \
    -method all -N 50 \
    --lsh_k 4 --lsh_L 15 --lsh_w 4000 \
    --hc_k 4 --hc_probes 50 --hc_w 4000 \
    --ivf_k 200 --ivf_probe 20

echo "=== Benchmarks Completed! Results are in '$LOG_DIR/' ==="