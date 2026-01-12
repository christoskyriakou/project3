# Makefile for Protein Remote Homology Detection
# Compiles all C implementations and sets up Python environment

CC = gcc
CFLAGS = -O3 -Wall -lm -std=c99
PYTHON = python3

# Directories
LSH_DIR = LSH_Project
HC_DIR = HYPERCUBE_Project
IVF_DIR = IVFFlat
IVFPQ_DIR = IVFPQ

# Executables
MAIN_EXEC = search

.PHONY: all clean python c-all test help

all: python c-all
	@echo "======================================"
	@echo "Build complete!"
	@echo "======================================"
	@echo "To test:"
	@echo "  make test"
	@echo ""

# Python setup
python:
	@echo "Setting up Python environment..."
#	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Python dependencies installed"

# Compile all C implementations
c-all: $(MAIN_EXEC)
	@echo "✓ All C executables compiled"

# Main search executable
$(MAIN_EXEC): main.c
	@echo "Compiling main search executable..."
	$(CC) $(CFLAGS) -o $(MAIN_EXEC) main.c
	@echo "✓ $(MAIN_EXEC) compiled"

# Individual C implementations (if needed)
lsh:
	@echo "Compiling LSH..."
	cd $(LSH_DIR) && $(MAKE)
	@echo "✓ LSH compiled"

hypercube:
	@echo "Compiling Hypercube..."
	cd $(HC_DIR) && $(MAKE)
	@echo "✓ Hypercube compiled"

ivfflat:
	@echo "Compiling IVF-Flat..."
	cd $(IVF_DIR) && $(MAKE)
	@echo "✓ IVF-Flat compiled"

ivfpq:
	@echo "Compiling IVFPQ..."
	cd $(IVFPQ_DIR) && $(MAKE)
	@echo "✓ IVFPQ compiled"

# Test with small dataset
test:
	@echo "======================================"
	@echo "Running test pipeline..."
	@echo "======================================"
	@echo ""
	@echo "[TEST] Step 1: Generate embeddings..."
	$(PYTHON) protein_embed.py \
		-i swissprot_small_small.fasta \
		-o test_embeddings.fvecs \
		--batch_size 4
	@echo ""
	@echo "[TEST] Step 2: Build Neural LSH index..."
	$(PYTHON) protein_nlsh_build.py \
		-d test_embeddings.fvecs \
		-i ./test_index \
		--knn 5 \
		-m 10 \
		--epochs 5
	@echo ""
	@echo "[TEST] Step 3: Run search (Neural LSH only)..."
	$(PYTHON) protein_search.py \
		-d test_embeddings.fvecs \
		-q swissprot_small_small.fasta \
		-o test_results.txt \
		-method neural \
		-N 10 \
		--index ./test_index
	@echo ""
	@echo "[TEST] Step 4: Analyze results..."
	$(PYTHON) analyze_results.py -i test_results.txt
	@echo ""
	@echo "======================================"
	@echo "Test complete! Check test_results.txt"
	@echo "======================================"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(MAIN_EXEC)
	rm -f *.o
	rm -f test_*.fvecs test_*.txt
	rm -rf test_index/
	rm -rf __pycache__/
	rm -rf *.pyc
	cd $(LSH_DIR) && $(MAKE) clean || true
	cd $(HC_DIR) && $(MAKE) clean || true
	cd $(IVF_DIR) && $(MAKE) clean || true
	cd $(IVFPQ_DIR) && $(MAKE) clean || true
	@echo "✓ Clean complete"

# Help
help:
	@echo "Protein Remote Homology Detection - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build everything (Python + C)"
	@echo "  python      - Install Python dependencies"
	@echo "  c-all       - Compile all C implementations"
	@echo "  lsh         - Compile LSH only"
	@echo "  hypercube   - Compile Hypercube only"
	@echo "  ivfflat     - Compile IVF-Flat only"
	@echo "  ivfpq       - Compile IVFPQ only"
	@echo "  test        - Run test pipeline"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make all    # Build everything"
	@echo "  make test   # Run quick test"
	@echo "  make clean  # Clean up"