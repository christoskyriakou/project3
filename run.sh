#!/bin/bash

# run_protein_search.sh
# Complete pipeline for protein remote homology detection

set -e

echo "======================================"
echo "Protein Remote Homology Detection"
echo "======================================"
echo ""

# Default parameters
DATA_FASTA="swissprot_small.fasta"
QUERY_FASTA="targets.fasta"
EMBEDDINGS_FILE="protein_vectors.fvecs"
INDEX_DIR="./protein_index"
OUTPUT_FILE="results.txt"
METHOD="all"
N=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_FASTA="$2"
            shift 2
            ;;
        --query)
            QUERY_FASTA="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        -N)
            N="$2"
            shift 2
            ;;
        --skip-embed)
            SKIP_EMBED=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Generate embeddings (if not skipped)
if [ -z "$SKIP_EMBED" ]; then
    echo "[1] Generating ESM-2 embeddings..."
    echo "    Input: $DATA_FASTA"
    echo "    Output: $EMBEDDINGS_FILE"
    echo ""
    
    python protein_embed.py \
        -i "$DATA_FASTA" \
        -o "$EMBEDDINGS_FILE" \
        --batch_size 8
    
    echo ""
    echo "[1] ✓ Embeddings generated"
    echo ""
else
    echo "[1] Skipping embedding generation (using existing $EMBEDDINGS_FILE)"
    echo ""
fi

# Step 2: Build Neural LSH index (if method is neural or all)
if [ "$METHOD" = "neural" ] || [ "$METHOD" = "all" ]; then
    if [ -z "$SKIP_BUILD" ]; then
        echo "[2] Building Neural LSH index..."
        echo "    Index directory: $INDEX_DIR"
        echo ""
        
        python protein_nlsh_build.py \
            -d "$EMBEDDINGS_FILE" \
            -i "$INDEX_DIR" \
            --knn 10 \
            -m 100 \
            --epochs 20 \
            --batch_size 128
        
        echo ""
        echo "[2] ✓ Neural LSH index built"
        echo ""
    else
        echo "[2] Skipping Neural LSH build (using existing index)"
        echo ""
    fi
fi

# Step 3: Run search benchmark
echo "[3] Running ANN search benchmark..."
echo "    Method: $METHOD"
echo "    Query: $QUERY_FASTA"
echo "    N: $N"
echo "    Output: $OUTPUT_FILE"
echo ""

python protein_search.py \
    -d "$EMBEDDINGS_FILE" \
    -q "$QUERY_FASTA" \
    -o "$OUTPUT_FILE" \
    -method "$METHOD" \
    -N "$N" \
    --index "$INDEX_DIR"

echo ""
echo "[3] ✓ Search complete"
echo ""

# Display summary
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_FILE"
echo ""
echo "To run analysis:"
echo "  python analyze_results.py -i $OUTPUT_FILE"
echo ""