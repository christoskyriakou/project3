#!/bin/bash


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROGRAM="./search"
RESULTS_DIR="experimental_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${RESULTS_DIR}/summary_${TIMESTAMP}.txt"
DETAILED_LOG="${RESULTS_DIR}/detailed_${TIMESTAMP}.log"

# Dataset files (CHANGE THESE TO YOUR ACTUAL FILES)
MNIST_INPUT="mnist_train.dat"
MNIST_QUERY="mnist_test.dat"
SIFT_INPUT="sift_base.fvecs"
SIFT_QUERY="sift_query.fvecs"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if program exists
check_program() {
    if [ ! -f "$PROGRAM" ]; then
        print_error "Program '$PROGRAM' not found!"
        print_info "Please compile first: make"
        exit 1
    fi
}

# Create results directory
setup_directories() {
    mkdir -p "$RESULTS_DIR"
    print_info "Results directory: $RESULTS_DIR"
}

# Run a single experiment
run_experiment() {
    local exp_name=$1
    local dataset_type=$2
    local input_file=$3
    local query_file=$4
    local kclusters=$5
    local nprobe=$6
    local N=$7
    local R=$8
    local seed=$9
    
    local output_file="${RESULTS_DIR}/${exp_name}_output.txt"
    local log_file="${RESULTS_DIR}/${exp_name}_log.txt"
    
    print_info "Running: $exp_name"
    
    # Check if input files exist
    if [ ! -f "$input_file" ]; then
        print_warning "Input file not found: $input_file (skipping)"
        echo "SKIPPED: $exp_name - Input file not found" >> "$SUMMARY_FILE"
        return 1
    fi
    
    if [ ! -f "$query_file" ]; then
        print_warning "Query file not found: $query_file (skipping)"
        echo "SKIPPED: $exp_name - Query file not found" >> "$SUMMARY_FILE"
        return 1
    fi
    
    # Run the experiment
    echo "========================================" >> "$DETAILED_LOG"
    echo "Experiment: $exp_name" >> "$DETAILED_LOG"
    echo "Started at: $(date)" >> "$DETAILED_LOG"
    echo "Parameters: type=$dataset_type, kclusters=$kclusters, nprobe=$nprobe, N=$N, R=$R, seed=$seed" >> "$DETAILED_LOG"
    echo "----------------------------------------" >> "$DETAILED_LOG"
    
    $PROGRAM -d "$input_file" -q "$query_file" -type "$dataset_type" \
             -kclusters "$kclusters" -nprobe "$nprobe" -N "$N" -R "$R" \
             -seed "$seed" -o "$output_file" -ivfflat -range true \
             > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_info "✓ Completed: $exp_name"
        
        # Extract key metrics from log
        echo "" >> "$SUMMARY_FILE"
        echo "=== $exp_name ===" >> "$SUMMARY_FILE"
        echo "Parameters: kclusters=$kclusters, nprobe=$nprobe, N=$N, R=$R, seed=$seed" >> "$SUMMARY_FILE"
        grep -E "(Silhouette|Average AF|Recall@|QPS|Avg approx time|Avg exact time|Speedup|Range speedup|R-near neighbors found)" "$log_file" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        
        # Append to detailed log
        cat "$log_file" >> "$DETAILED_LOG"
        echo "" >> "$DETAILED_LOG"
        
        return 0
    else
        print_error "✗ Failed: $exp_name (exit code: $exit_code)"
        echo "FAILED: $exp_name (exit code: $exit_code)" >> "$SUMMARY_FILE"
        echo "ERROR: Exit code $exit_code" >> "$DETAILED_LOG"
        echo "" >> "$DETAILED_LOG"
        return 1
    fi
}

################################################################################
# Experiment Configurations
################################################################################

# Experiment 1: Effect of k-clusters (MNIST)
run_kclusters_experiment_mnist() {
    print_header "Experiment 1: Effect of k-clusters (MNIST)"
    
    local N=10
    local R=2000
    local seed=1
    
    for kclusters in 20 30 50 75 100; do
        local nprobe=$((kclusters / 10))  # nprobe = 10% of kclusters
        if [ $nprobe -lt 3 ]; then nprobe=3; fi
        
        run_experiment "mnist_kclusters_${kclusters}" "mnist" \
                      "$MNIST_INPUT" "$MNIST_QUERY" \
                      "$kclusters" "$nprobe" "$N" "$R" "$seed"
    done
}

# Experiment 2: Effect of nprobe (MNIST)
run_nprobe_experiment_mnist() {
    print_header "Experiment 2: Effect of nprobe (MNIST)"
    
    local kclusters=50
    local N=10
    local R=2000
    local seed=1
    
    for nprobe in 1 3 5 10 15 20; do
        run_experiment "mnist_nprobe_${nprobe}" "mnist" \
                      "$MNIST_INPUT" "$MNIST_QUERY" \
                      "$kclusters" "$nprobe" "$N" "$R" "$seed"
    done
}

# Experiment 3: Effect of N (number of neighbors) (MNIST)
run_N_experiment_mnist() {
    print_header "Experiment 3: Effect of N (MNIST)"
    
    local kclusters=50
    local nprobe=5
    local R=2000
    local seed=1
    
    for N in 1 5 10 20 50 100; do
        run_experiment "mnist_N_${N}" "mnist" \
                      "$MNIST_INPUT" "$MNIST_QUERY" \
                      "$kclusters" "$nprobe" "$N" "$R" "$seed"
    done
}

# Experiment 4: Effect of seed (MNIST)
run_seed_experiment_mnist() {
    print_header "Experiment 4: Effect of seed (MNIST)"
    
    local kclusters=50
    local nprobe=5
    local N=10
    local R=2000
    
    for seed in 1 42 123 999 2025; do
        run_experiment "mnist_seed_${seed}" "mnist" \
                      "$MNIST_INPUT" "$MNIST_QUERY" \
                      "$kclusters" "$nprobe" "$N" "$R" "$seed"
    done
}

# Experiment 5: Effect of R (radius) (MNIST)
run_radius_experiment_mnist() {
    print_header "Experiment 5: Effect of R (MNIST)"
    
    local kclusters=50
    local nprobe=5
    local N=10
    local seed=1
    
    for R in 500 1000 1500 2000 3000 5000; do
        run_experiment "mnist_R_${R}" "mnist" \
                      "$MNIST_INPUT" "$MNIST_QUERY" \
                      "$kclusters" "$nprobe" "$N" "$R" "$seed"
    done
}

# Experiment 6: Effect of k-clusters (SIFT)
run_kclusters_experiment_sift() {
    print_header "Experiment 6: Effect of k-clusters (SIFT)"
    
    local N=10
    local R=200
    local seed=1
    
    for kclusters in 50 100 150 200 300; do
        local nprobe=$((kclusters / 10))
        if [ $nprobe -lt 5 ]; then nprobe=5; fi
        
        run_experiment "sift_kclusters_${kclusters}" "sift" \
                      "$SIFT_INPUT" "$SIFT_QUERY" \
                      "$kclusters" "$nprobe" "$N" "$R" "$seed"
    done
}

# Experiment 7: Effect of nprobe (SIFT)
run_nprobe_experiment_sift() {
    print_header "Experiment 7: Effect of nprobe (SIFT)"
    
    local kclusters=100
    local N=10
    local R=200
    local seed=1
    
    for nprobe in 1 5 10 20 30 50; do
        run_experiment "sift_nprobe_${nprobe}" "sift" \
                      "$SIFT_INPUT" "$SIFT_QUERY" \
                      "$kclusters" "$nprobe" "$N" "$R" "$seed"
    done
}

# Experiment 8: Comparison at different configurations (MNIST)
run_comparison_experiment_mnist() {
    print_header "Experiment 8: Configuration Comparison (MNIST)"
    
    local seed=1
    local N=10
    local R=2000
    
    # Fast configuration
    run_experiment "mnist_config_fast" "mnist" \
                  "$MNIST_INPUT" "$MNIST_QUERY" \
                  "20" "2" "$N" "$R" "$seed"
    
    # Balanced configuration
    run_experiment "mnist_config_balanced" "mnist" \
                  "$MNIST_INPUT" "$MNIST_QUERY" \
                  "50" "5" "$N" "$R" "$seed"
    
    # Accurate configuration
    run_experiment "mnist_config_accurate" "mnist" \
                  "$MNIST_INPUT" "$MNIST_QUERY" \
                  "100" "20" "$N" "$R" "$seed"
}

# Experiment 9: Comparison at different configurations (SIFT)
run_comparison_experiment_sift() {
    print_header "Experiment 9: Configuration Comparison (SIFT)"
    
    local seed=1
    local N=10
    local R=200
    
    # Fast configuration
    run_experiment "sift_config_fast" "sift" \
                  "$SIFT_INPUT" "$SIFT_QUERY" \
                  "50" "5" "$N" "$R" "$seed"
    
    # Balanced configuration
    run_experiment "sift_config_balanced" "sift" \
                  "$SIFT_INPUT" "$SIFT_QUERY" \
                  "100" "10" "$N" "$R" "$seed"
    
    # Accurate configuration
    run_experiment "sift_config_accurate" "sift" \
                  "$SIFT_INPUT" "$SIFT_QUERY" \
                  "200" "30" "$N" "$R" "$seed"
}

################################################################################
# Main Execution
################################################################################

main() {
    print_header "IVFFlat Experimental Study"
    echo "Started at: $(date)"
    echo ""
    
    # Setup
    check_program
    setup_directories
    
    # Initialize summary file
    echo "IVFFlat Experimental Study Results" > "$SUMMARY_FILE"
    echo "Generated at: $(date)" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Initialize detailed log
    echo "IVFFlat Experimental Study - Detailed Log" > "$DETAILED_LOG"
    echo "Generated at: $(date)" >> "$DETAILED_LOG"
    echo "========================================" >> "$DETAILED_LOG"
    echo "" >> "$DETAILED_LOG"
    
    # Run experiments based on command line argument
    case "${1:-all}" in
        "mnist")
            print_info "Running MNIST experiments only"
            run_kclusters_experiment_mnist
            run_nprobe_experiment_mnist
            run_N_experiment_mnist
            run_seed_experiment_mnist
            run_radius_experiment_mnist
            run_comparison_experiment_mnist
            ;;
        "sift")
            print_info "Running SIFT experiments only"
            run_kclusters_experiment_sift
            run_nprobe_experiment_sift
            run_comparison_experiment_sift
            ;;
        "quick")
            print_info "Running quick test experiments"
            run_nprobe_experiment_mnist
            ;;
        "all")
            print_info "Running all experiments"
            run_kclusters_experiment_mnist
            run_nprobe_experiment_mnist
            run_N_experiment_mnist
            run_seed_experiment_mnist
            run_radius_experiment_mnist
            run_comparison_experiment_mnist
            
            run_kclusters_experiment_sift
            run_nprobe_experiment_sift
            run_comparison_experiment_sift
            ;;
        *)
            print_error "Unknown experiment type: $1"
            echo "Usage: $0 [all|mnist|sift|quick]"
            echo "  all   - Run all experiments (default)"
            echo "  mnist - Run MNIST experiments only"
            echo "  sift  - Run SIFT experiments only"
            echo "  quick - Run quick test"
            exit 1
            ;;
    esac
    
    # Finalize
    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "Experiments completed at: $(date)" >> "$SUMMARY_FILE"
    
    print_header "Experiments Completed!"
    print_info "Summary file: $SUMMARY_FILE"
    print_info "Detailed log: $DETAILED_LOG"
    print_info "Individual outputs: $RESULTS_DIR/*_output.txt"
    
    echo ""
    print_info "To view summary:"
    echo "  cat $SUMMARY_FILE"
    echo ""
}

# Run main function
main "$@"