

set -e  # Exit on error

# ============================================
# CONFIGURATION
# ============================================

EXECUTABLE="./search"
MNIST_INPUT="res/train-images.idx3-ubyte"
MNIST_QUERY="res/t10k-images.idx3-ubyte"
SIFT_INPUT="res/sift_base.fvecs"
SIFT_QUERY="res/sift_query.fvecs"

# Output directory
RESULTS_DIR="experiment_results"
LOG_FILE="$RESULTS_DIR/full_experiment_log.txt"
SUMMARY_FILE="$RESULTS_DIR/experiment_summary.csv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# SETUP
# ============================================

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   IVFPQ Complete Experiment Suite     ${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: Executable '$EXECUTABLE' not found!${NC}"
    echo "Please compile first: make clean && make"
    exit 1
fi

# Initialize log file
cat > "$LOG_FILE" << EOF
IVFPQ Experimental Study - Complete Log
========================================
Date: $(date)
Host: $(hostname)
Start Time: $(date +%s)

EOF

# Initialize CSV summary
cat > "$SUMMARY_FILE" << EOF
Experiment,Dataset,kclusters,nprobe,M,nbits,N,Recall,QPS,AF,Speedup,Silhouette,TrainingTime
EOF

# ============================================
# HELPER FUNCTIONS
# ============================================

run_experiment() {
    local exp_name=$1
    local dataset_type=$2
    local input_file=$3
    local query_file=$4
    local kclusters=$5
    local nprobe=$6
    local M=$7
    local nbits=$8
    local N=$9
    
    local output_file="$RESULTS_DIR/${exp_name}.txt"
    local console_log="$RESULTS_DIR/${exp_name}_console.log"
    
    echo -e "${BLUE}Running: $exp_name${NC}"
    echo "  Dataset: $dataset_type, k=$kclusters, nprobe=$nprobe, M=$M, nbits=$nbits, N=$N"
    
    # Log to main file
    cat >> "$LOG_FILE" << EOF

================================================================================
Experiment: $exp_name
================================================================================
Dataset Type: $dataset_type
Parameters: kclusters=$kclusters, nprobe=$nprobe, M=$M, nbits=$nbits, N=$N
Command: $EXECUTABLE -d $input_file -q $query_file -o $output_file \\
         -type $dataset_type -kclusters $kclusters -nprobe $nprobe \\
         -M $M -nbits $nbits -N $N -R 2.0 -ivfpq
Start Time: $(date)

Console Output:
-------------------------------------------------------------------------------
EOF
    
    # Run experiment and capture output
    local start_time=$(date +%s)
    
    if $EXECUTABLE -d "$input_file" -q "$query_file" -o "$output_file" \
                    -type "$dataset_type" -kclusters "$kclusters" -nprobe "$nprobe" \
                    -M "$M" -nbits "$nbits" -N "$N" -R 2.0 -ivfpq \
                    2>&1 | tee "$console_log" >> "$LOG_FILE"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Extract metrics from console output
        local recall=$(grep "Recall@" "$console_log" | tail -1 | awk '{print $2}')
        local qps=$(grep "QPS:" "$console_log" | tail -1 | awk '{print $2}')
        local af=$(grep "Average Approximation Factor:" "$console_log" | tail -1 | awk '{print $4}')
        local speedup=$(grep "Speedup:" "$console_log" | tail -1 | awk '{print $2}' | tr -d 'x')
        local silhouette=$(grep "Silhouette Score:" "$console_log" | tail -1 | awk '{print $3}')
        local training=$(grep "Index built in" "$console_log" | awk '{print $4}')
        
        # Append to CSV
        echo "$exp_name,$dataset_type,$kclusters,$nprobe,$M,$nbits,$N,$recall,$qps,$af,$speedup,$silhouette,$training" >> "$SUMMARY_FILE"
        
        echo -e "${GREEN}✓ Completed in ${duration}s${NC}"
        echo "  Recall=$recall, QPS=$qps, AF=$af, Speedup=${speedup}x"
        
        # Log completion
        cat >> "$LOG_FILE" << EOF

-------------------------------------------------------------------------------
End Time: $(date)
Duration: ${duration} seconds
Status: SUCCESS
Metrics: Recall=$recall, QPS=$qps, AF=$af, Speedup=${speedup}x, Silhouette=$silhouette

EOF
        
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo -e "${RED}✗ Failed after ${duration}s${NC}"
        
        cat >> "$LOG_FILE" << EOF

-------------------------------------------------------------------------------
End Time: $(date)
Duration: ${duration} seconds
Status: FAILED

EOF
        
        return 1
    fi
}

# ============================================
# EXPERIMENT 1: nprobe Variation (SIFT)
# ============================================

echo -e "\n${YELLOW}=== Experiment 1: nprobe Variation (SIFT) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 1: nprobe Variation (SIFT)
################################################################################
Objective: Analyze trade-off between accuracy and speed
Fixed: k=50, M=8, nbits=8, N=10
Variable: nprobe ∈ {1, 2, 5, 10, 20, 30}

EOF

NPROBE_VALUES=(1 2 5 10 20 30)
for nprobe in "${NPROBE_VALUES[@]}"; do
    run_experiment "exp1_sift_nprobe_${nprobe}" "sift" \
                   "$SIFT_INPUT" "$SIFT_QUERY" \
                   50 "$nprobe" 8 8 10 || true
    sleep 2
done

# ============================================
# EXPERIMENT 2: nprobe Variation (MNIST)
# ============================================

echo -e "\n${YELLOW}=== Experiment 2: nprobe Variation (MNIST) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 2: nprobe Variation (MNIST)
################################################################################
Objective: Analyze trade-off between accuracy and speed
Fixed: k=50, M=16, nbits=8, N=10
Variable: nprobe ∈ {1, 2, 5, 10, 20}

EOF

NPROBE_VALUES=(1 2 5 10 20)
for nprobe in "${NPROBE_VALUES[@]}"; do
    run_experiment "exp2_mnist_nprobe_${nprobe}" "mnist" \
                   "$MNIST_INPUT" "$MNIST_QUERY" \
                   50 "$nprobe" 16 8 10 || true
    sleep 2
done

# ============================================
# EXPERIMENT 3: M Variation (SIFT)
# ============================================

echo -e "\n${YELLOW}=== Experiment 3: M Variation (SIFT) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 3: M Variation (SIFT)
################################################################################
Objective: Effect of Product Quantization granularity
Fixed: k=50, nprobe=10, nbits=8, N=10
Variable: M ∈ {4, 8, 16, 32} (must divide 128)

EOF

M_VALUES=(4 8 16 32)
for M in "${M_VALUES[@]}"; do
    run_experiment "exp3_sift_M_${M}" "sift" \
                   "$SIFT_INPUT" "$SIFT_QUERY" \
                   50 10 "$M" 8 10 || true
    sleep 2
done

# ============================================
# EXPERIMENT 4: M Variation (MNIST)
# ============================================

echo -e "\n${YELLOW}=== Experiment 4: M Variation (MNIST) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 4: M Variation (MNIST)
################################################################################
Objective: Effect of Product Quantization granularity
Fixed: k=50, nprobe=10, nbits=8, N=10
Variable: M ∈ {4, 7, 8, 16, 49} (must divide 784)

EOF

M_VALUES=(4 7 8 16 49)
for M in "${M_VALUES[@]}"; do
    run_experiment "exp4_mnist_M_${M}" "mnist" \
                   "$MNIST_INPUT" "$MNIST_QUERY" \
                   50 10 "$M" 8 10 || true
    sleep 2
done

# ============================================
# EXPERIMENT 5: kclusters Variation (SIFT)
# ============================================

echo -e "\n${YELLOW}=== Experiment 5: kclusters Variation (SIFT) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 5: kclusters Variation (SIFT)
################################################################################
Objective: Effect of coarse quantization resolution
Fixed: nprobe=10, M=8, nbits=8, N=10
Variable: kclusters ∈ {20, 30, 50, 80, 100}

EOF

K_VALUES=(20 30 50 80 100)
for k in "${K_VALUES[@]}"; do
    run_experiment "exp5_sift_k_${k}" "sift" \
                   "$SIFT_INPUT" "$SIFT_QUERY" \
                   "$k" 10 8 8 10 || true
    sleep 2
done

# ============================================
# EXPERIMENT 6: kclusters Variation (MNIST)
# ============================================

echo -e "\n${YELLOW}=== Experiment 6: kclusters Variation (MNIST) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 6: kclusters Variation (MNIST)
################################################################################
Objective: Effect of coarse quantization resolution
Fixed: nprobe=10, M=16, nbits=8, N=10
Variable: kclusters ∈ {20, 30, 50, 80, 100}

EOF

K_VALUES=(20 30 50 80 100)
for k in "${K_VALUES[@]}"; do
    run_experiment "exp6_mnist_k_${k}" "mnist" \
                   "$MNIST_INPUT" "$MNIST_QUERY" \
                   "$k" 10 16 8 10 || true
    sleep 2
done

# ============================================
# EXPERIMENT 7: nbits Variation (SIFT)
# ============================================

echo -e "\n${YELLOW}=== Experiment 7: nbits Variation (SIFT) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 7: nbits Variation (SIFT)
################################################################################
Objective: Effect of PQ codebook size
Fixed: k=50, nprobe=10, M=8, N=10
Variable: nbits ∈ {4, 6, 8}
Note: nbits=10 may fail with small datasets (n<10K)

EOF

NBITS_VALUES=(4 6 8)
for nbits in "${NBITS_VALUES[@]}"; do
    run_experiment "exp7_sift_nbits_${nbits}" "sift" \
                   "$SIFT_INPUT" "$SIFT_QUERY" \
                   50 10 8 "$nbits" 10 || true
    sleep 2
done

# ============================================
# EXPERIMENT 8: nbits Variation (MNIST)
# ============================================

echo -e "\n${YELLOW}=== Experiment 8: nbits Variation (MNIST) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 8: nbits Variation (MNIST)
################################################################################
Objective: Effect of PQ codebook size
Fixed: k=50, nprobe=10, M=16, N=10
Variable: nbits ∈ {4, 6, 8}

EOF

NBITS_VALUES=(4 6 8)
for nbits in "${NBITS_VALUES[@]}"; do
    run_experiment "exp8_mnist_nbits_${nbits}" "mnist" \
                   "$MNIST_INPUT" "$MNIST_QUERY" \
                   50 10 16 "$nbits" 10 || true
    sleep 2
done

# ============================================
# EXPERIMENT 9: N Variation (SIFT)
# ============================================

echo -e "\n${YELLOW}=== Experiment 9: N Variation (SIFT) ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 9: N Variation (SIFT)
################################################################################
Objective: Effect of number of neighbors retrieved
Fixed: k=50, nprobe=10, M=8, nbits=8
Variable: N ∈ {1, 5, 10, 20, 50}

EOF

N_VALUES=(1 5 10 20 50)
for N in "${N_VALUES[@]}"; do
    run_experiment "exp9_sift_N_${N}" "sift" \
                   "$SIFT_INPUT" "$SIFT_QUERY" \
                   50 10 8 8 "$N" || true
    sleep 2
done

# ============================================
# EXPERIMENT 10: Optimal Configurations
# ============================================

echo -e "\n${YELLOW}=== Experiment 10: Optimal Configurations ===${NC}\n"
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT 10: Optimal Configurations
################################################################################
Objective: Test recommended configurations for different scenarios

EOF

# SIFT - Speed focused
run_experiment "exp10_sift_speed" "sift" \
               "$SIFT_INPUT" "$SIFT_QUERY" \
               25 1 8 8 10 || true
sleep 2

# SIFT - Balanced
run_experiment "exp10_sift_balanced" "sift" \
               "$SIFT_INPUT" "$SIFT_QUERY" \
               50 5 8 8 10 || true
sleep 2

# SIFT - Quality focused
run_experiment "exp10_sift_quality" "sift" \
               "$SIFT_INPUT" "$SIFT_QUERY" \
               50 20 32 8 10 || true
sleep 2

# MNIST - Speed focused
run_experiment "exp10_mnist_speed" "mnist" \
               "$MNIST_INPUT" "$MNIST_QUERY" \
               25 2 8 8 10 || true
sleep 2

# MNIST - Balanced
run_experiment "exp10_mnist_balanced" "mnist" \
               "$MNIST_INPUT" "$MNIST_QUERY" \
               50 5 16 8 10 || true
sleep 2

# MNIST - Quality focused
run_experiment "exp10_mnist_quality" "mnist" \
               "$MNIST_INPUT" "$MNIST_QUERY" \
               50 10 32 8 10 || true
sleep 2

# ============================================
# FINALIZATION
# ============================================

# Calculate total time
END_TIME=$(date +%s)
cat >> "$LOG_FILE" << EOF

################################################################################
EXPERIMENT SUITE COMPLETED
################################################################################
End Time: $(date)
Total Duration: $((END_TIME - $(grep "Start Time:" "$LOG_FILE" | head -1 | awk '{print $3}'))) seconds

All results saved in: $RESULTS_DIR/
- Full log: $LOG_FILE
- CSV summary: $SUMMARY_FILE
- Individual outputs: $RESULTS_DIR/exp*.txt
- Console logs: $RESULTS_DIR/exp*_console.log

EOF

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   Experiments Completed Successfully   ${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "Results saved in: $RESULTS_DIR/"
echo ""
