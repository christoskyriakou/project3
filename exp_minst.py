#!/usr/bin/env python3
"""
Πειραματική Μελέτη - Neural LSH v2
Υποστηρίζει sift_learn για γρηγορότερο training
"""

import subprocess
import os
import json
import time
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    'sift': {
        'train': 'data/sift/sift_learn.fvecs',      # 100K για training
        'base': 'data/sift/sift_learn.fvecs',        # 1M για search
        'query': 'data/sift/sift_query.fvecs',
        'type': 'sift',
        # 'default_R': 2800
    },
    'sift_full': {
        'train': 'data/sift/sift_base.fvecs',       # 1M για training (slow)
        'base': 'data/sift/sift_base.fvecs',
        'query': 'data/sift/sift_query.fvecs',
        'type': 'sift',
        'default_R': 2800
    },
    'mnist': {
        'train': 'data/mnist/train-images.idx3-ubyte',  # 60K
        'base': 'data/mnist/train-images.idx3-ubyte',
        'query': 'data/mnist/t10k-images.idx3-ubyte',
        'type': 'mnist',
        'default_R': 2000
    }
}

# Υπερπαράμετροι για testing mnist
# EXPERIMENTS = {
#     'baseline': {
#         'knn': 10,
#         'm': 100,
#         'epochs': 10,
#         'layers': 3,
#         'nodes': 64,
#         'batch_size': 128,
#         'lr': 0.001,
#         'imbalance': 0.03,
#         'T': 5,
#         'description': 'Quick baseline με default params'
#     },
#     'fewer_bins': {
#         'knn': 10,
#         'm': 50,
#         'epochs': 15,
#         'layers': 3,
#         'nodes': 64,
#         'batch_size': 256,
#         'lr': 0.001,
#         'imbalance': 0.03,
#         'T': 10,
#         'description': 'Λιγότερα bins (m=50) για μεγαλύτερα partitions'
#     },
#     'deeper_net': {
#         'knn': 15,
#         'm': 50,
#         'epochs': 20,
#         'layers': 5,
#         'nodes': 128,
#         'batch_size': 256,
#         'lr': 0.0008,
#         'imbalance': 0.04,
#         'T': 10,
#         'description': 'Βαθύτερο MLP (5 layers, 128 nodes)'
#     },
#     'optimal': {
#         'knn': 20,
#         'm': 30,
#         'epochs': 30,
#         'layers': 4,
#         'nodes': 128,
#         'batch_size': 256,
#         'lr': 0.0005,
#         'imbalance': 0.05,
#         'T': 10,
#         'description': 'Βέλτιστο balance recall/QPS'
#     },
#     'high_recall': {
#         'knn': 25,
#         'm': 25,
#         'epochs': 40,
#         'layers': 4,
#         'nodes': 128,
#         'batch_size': 256,
#         'lr': 0.0005,
#         'imbalance': 0.05,
#         'T': 12,
#         'description': 'Μέγιστο recall (αργότερο)'
#     },
#     'fast_search': {
#         'knn': 10,
#         'm': 80,
#         'epochs': 15,
#         'layers': 3,
#         'nodes': 64,
#         'batch_size': 256,
#         'lr': 0.001,
#         'imbalance': 0.03,
#         'T': 5,
#         'description': 'Γρήγορο search, χαμηλότερο recall'
#     }
# }
# SIFT
EXPERIMENTS = {
    'baseline': {
        'knn': 15,
        'm': 100,
        'epochs': 20,
        'layers': 3,
        'nodes': 64,
        'batch_size': 128,
        'lr': 0.001,
        'imbalance': 0.03,
        'T': 20,
        'description': 'Quick baseline με default params'
    },
    'fewer_bins': {
        'knn': 15,
        'm': 40,
        'epochs': 20,
        'layers': 4,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.001,
        'imbalance': 0.04,
        'T': 12,
        'description': 'Λιγότερα bins (m=40) για μεγαλύτερα partitions'
    },
    'deeper_net': {
        'knn': 20,
        'm': 35,
        'epochs': 25,
        'layers': 5,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.0008,
        'imbalance': 0.05,
        'T': 14,
        'description': 'Βαθύτερο MLP (5 layers, 128 nodes)'
    },
    'optimal': {
        'knn': 25,
        'm': 30,
        'epochs': 35,
        'layers': 4,
        'nodes': 128,
        'batch_size': 256,
        'lr': 0.0005,
        'imbalance': 0.05,
        'T': 12,
        'description': 'Βέλτιστο balance recall/QPS'
    },
    'high_recall': {
        'knn': 40,            # ← Πολύ πλούσιος γράφος
        'm': 15,              # ← Πολύ λίγα, τεράστια bins
        'epochs': 70,
        'layers': 6,
        'nodes': 320,         # ← Περισσότερη capacity
        'batch_size': 256,
        'lr': 0.00015,        # ← Πολύ προσεκτικό
        'imbalance': 0.12,    # ← Πολύ χαλαρό
        'T': 10,              # ← 67% των bins
        'description': 'Target 70-75% - Very high quality'
    },
    'fast_search': {
        'knn': 10,
        'm': 80,
        'epochs': 15,
        'layers': 3,
        'nodes': 64,
        'batch_size': 256,
        'lr': 0.001,
        'imbalance': 0.03,
        'T': 6,
        'description': 'Γρήγορο search, χαμηλότερο recall'
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_command(cmd, desc="", verbose=True):
    """Εκτελεί command και εμφανίζει output"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 {desc}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if verbose:
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        if not verbose:
            print(result.stdout)
            print(result.stderr)
        return None, elapsed
    
    if verbose:
        print(f"✅ Completed in {elapsed:.2f}s")
    
    return result, elapsed


def parse_results(output_file):
    """Διαβάζει το output file και εξάγει metrics"""
    metrics = {}
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith("Average AF:"):
                metrics['avg_af'] = float(line.split(":")[1].strip())
            elif line.startswith("Recall@"):
                metrics['recall'] = float(line.split(":")[1].strip())
            elif line.startswith("QPS:"):
                metrics['qps'] = float(line.split(":")[1].strip())
            elif line.startswith("tApproximateAverage:"):
                metrics['t_approx'] = float(line.split(":")[1].strip())
            elif line.startswith("tTrueAverage:"):
                metrics['t_true'] = float(line.split(":")[1].strip())
    except Exception as e:
        print(f"⚠️  Error parsing {output_file}: {e}")
        return None
    
    return metrics


def check_dataset_exists(dataset_config):
    """Ελέγχει αν υπάρχουν τα αρχεία dataset"""
    missing = []
    
    for key in ['train', 'base', 'query']:
        if key in dataset_config:
            path = dataset_config[key]
            if not os.path.exists(path):
                missing.append(path)
    
    return missing


# ============================================================
# BUILD INDEX (WITH SEPARATE TRAIN/BASE SUPPORT)
# ============================================================

def build_index(dataset_name, experiment_name, params, use_improved=False):
    """
    Χτίζει index για συγκεκριμένο dataset και παραμέτρους
    
    Αν train != base (π.χ. sift_learn), χτίζει το index με το train set
    και μετά το μεταφέρει στο base set για search
    """
    
    dataset = DATASETS[dataset_name]
    index_path = f"indexes/sift_kahip0mine/{dataset_name}_{experiment_name}"
    
    os.makedirs(index_path, exist_ok=True)
    
    # Use train dataset for building
    train_data = dataset['train']
    
    build_script = "nlsh_build.py" if not use_improved else "nlsh_build.py"
    
    cmd = [
        "python3", build_script,
        "-d", train_data,
        "-i", index_path,
        "-type", dataset['type'],
        "--knn", str(params['knn']),
        "-m", str(params['m']),
        "--epochs", str(params['epochs']),
        "--layers", str(params['layers']),
        "--nodes", str(params['nodes']),
        "--batch_size", str(params['batch_size']),
        "--lr", str(params['lr']),
        "--imbalance", str(params['imbalance'])
    ]
    
    desc = f"Building index: {dataset_name} - {experiment_name}"
    result, build_time = run_command(cmd, desc)
    
    if result is None:
        return None, build_time
    
    # If train != base, need to re-index base data with trained model
    if dataset['train'] != dataset['base']:
        print(f"\n⚙️  Re-indexing {dataset['base']} with trained model...")
        reindex_time = reindex_with_trained_model(
            dataset['base'], 
            index_path, 
            dataset['type'],
            params
        )
        build_time += reindex_time
    
    return index_path, build_time


def reindex_with_trained_model(base_data, index_path, data_type, params):
    """
    Φορτώνει το trained μοντέλο και το τρέχει στο base dataset
    για να δημιουργήσει το inverted index
    """
    
    import torch
    import numpy as np
    from models import MLP
    from dataset_parser import load_dataset
    
    start_time = time.time()
    
    print(f"[REINDEX] Loading base dataset: {base_data}")
    X_base, _ = load_dataset(base_data)
    n, d = X_base.shape
    
    print(f"[REINDEX] Loading trained model from {index_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = os.path.join(index_path, "model.pth")
    model = MLP(
        d_in=d,
        num_classes=params['m'],
        layers=params['layers'],
        hidden=params['nodes']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"[REINDEX] Predicting partitions for {n} vectors...")
    
    # Predict partitions in batches
    batch_size = 1024
    predictions = []
    
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X_base[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            logits = model(batch_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
    
    predictions = np.array(predictions, dtype=np.int32)
    
    print(f"[REINDEX] Building new inverted index...")
    inverted = {r: [] for r in range(params['m'])}
    for idx, r in enumerate(predictions):
        inverted[int(r)].append(int(idx))
    
    # Save new inverted index
    inv_path = os.path.join(index_path, "inverted_index.npy")
    np.save(inv_path, np.array(inverted, dtype=object))
    
    # Also save the predictions
    parts_path = os.path.join(index_path, "partitions_base.npy")
    np.save(parts_path, predictions)
    
    elapsed = time.time() - start_time
    print(f"[REINDEX] Completed in {elapsed:.2f}s")
    
    # Print statistics
    sizes = [len(inverted[r]) for r in range(params['m'])]
    print(f"[REINDEX] Index stats: {params['m']} bins, avg={np.mean(sizes):.1f}, min={min(sizes)}, max={max(sizes)}")
    
    return elapsed


# ============================================================
# SEARCH
# ============================================================

def run_search(dataset_name, experiment_name, index_path, params, N=10):
    """Εκτελεί search για συγκεκριμένο index"""
    
    dataset = DATASETS[dataset_name]
    output_dir = f"results/siftkahip0mine/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{experiment_name}_N{N}.txt"
    
    # Use base dataset for search (not train)
    base_data = dataset['base']
    
    cmd = [
        "python3", "nlsh_search.py",
        "-d", base_data,
        "-q", dataset['query'],
        "-i", index_path,
        "-o", output_file,
        "-type", dataset['type'],
        "-N", str(N),
        "-T", str(params['T']),
        # "-R", str(dataset['default_R']),
        "--layers", str(params['layers']),
        "--nodes", str(params['nodes'])
    ]
    
    desc = f"Searching: {dataset_name} - {experiment_name} (N={N})"
    result, search_time = run_command(cmd, desc)
    
    # Parse results
    metrics = parse_results(output_file)
    
    return metrics, search_time


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_all_experiments(datasets=['sift', 'mnist'], 
                       experiments=None, 
                       N_values=[1, 5, 10],
                       use_improved=False):
    """Εκτελεί όλα τα experiments"""
    
    if experiments is None:
        experiments = list(EXPERIMENTS.keys())
    
    # Check datasets exist
    print("\n🔍 Checking datasets...")
    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            print(f"❌ Unknown dataset: {dataset_name}")
            print(f"Available: {list(DATASETS.keys())}")
            return None, None
        
        missing = check_dataset_exists(DATASETS[dataset_name])
        if missing:
            print(f"❌ Missing files for {dataset_name}:")
            for path in missing:
                print(f"   - {path}")
            print(f"\nSkipping {dataset_name}")
            datasets.remove(dataset_name)
    
    if not datasets:
        print("❌ No valid datasets found!")
        return None, None
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'#'*60}")
    print(f"# ΠΕΙΡΑΜΑΤΙΚΗ ΜΕΛΕΤΗ - Neural LSH v2")
    print(f"# Timestamp: {timestamp}")
    print(f"# Datasets: {datasets}")
    print(f"# Experiments: {experiments}")
    print(f"# N values: {N_values}")
    print(f"{'#'*60}\n")
    
    total_start = time.time()
    
    for dataset_name in datasets:
        results[dataset_name] = {}
        
        print(f"\n{'='*60}")
        print(f"📊 DATASET: {dataset_name.upper()}")
        
        # Show dataset info
        ds = DATASETS[dataset_name]
        if ds['train'] != ds['base']:
            print(f"   Training on: {ds['train']}")
            print(f"   Searching in: {ds['base']}")
        else:
            print(f"   Dataset: {ds['base']}")
        
        print(f"{'='*60}")
        
        for exp_name in experiments:
            print(f"\n{'─'*60}")
            print(f"🧪 Experiment: {exp_name}")
            
            params = EXPERIMENTS[exp_name]
            print(f"   Description: {params['description']}")
            print(f"   Parameters: k={params['knn']}, m={params['m']}, epochs={params['epochs']}, T={params['T']}")
            print(f"{'─'*60}")
            
            results[dataset_name][exp_name] = {
                'params': params,
                'N_results': {}
            }
            
            # Build index
            index_path, build_time = build_index(dataset_name, exp_name, params, use_improved)
            # build_time = 6000
            # index_path = f"indexes/sift_kahip0mine/{dataset_name}_{exp_name}"
            results[dataset_name][exp_name]['build_time'] = build_time
            
            if index_path is None:
                print(f"⚠️  Skipping search for {exp_name} due to build failure")
                continue
            
            # Run searches for different N values
            for N in N_values:
                metrics, search_time = run_search(dataset_name, exp_name, index_path, params, N)
                
                if metrics:
                    results[dataset_name][exp_name]['N_results'][N] = {
                        'metrics': metrics,
                        'search_time': search_time
                    }
                    
                    print(f"\n   📈 Results for N={N}:")
                    print(f"      Recall@{N}: {metrics.get('recall', 0)*100:.2f}%")
                    print(f"      Avg AF: {metrics.get('avg_af', 0):.4f}")
                    print(f"      QPS: {metrics.get('qps', 0):.2f}")
                    print(f"      Time: {metrics.get('t_approx', 0)*1000:.2f}ms")
    
    total_time = time.time() - total_start
    
    # Save results to JSON
    results_file = f"results/experiments_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ All experiments completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📁 Results saved to: {results_file}")
    print(f"{'='*60}\n")
    
    return results, results_file


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_markdown_report(results, results_file):
    """Δημιουργεί markdown report με τα αποτελέσματα"""
    
    report_file = results_file.replace('.json', '_report.md')
    
    with open(report_file, 'w') as f:
        f.write("# Πειραματική Μελέτη - Neural LSH\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall comparison
        f.write("## Summary - All Datasets\n\n")
        f.write("| Dataset | Experiment | Recall@10 | Avg AF | QPS | Build Time (min) |\n")
        f.write("|---------|------------|-----------|--------|-----|------------------|\n")
        
        for dataset_name, dataset_results in results.items():
            for exp_name, exp_data in dataset_results.items():
                build_time = exp_data.get('build_time', 0) / 60
                
                if 10 in exp_data.get('N_results', {}):
                    metrics = exp_data['N_results'][10]['metrics']
                    recall = metrics.get('recall', 0) * 100
                    af = metrics.get('avg_af', 0)
                    qps = metrics.get('qps', 0)
                    
                    f.write(f"| {dataset_name} | {exp_name} | {recall:.2f}% | {af:.4f} | {qps:.2f} | {build_time:.2f} |\n")
        
        f.write("\n---\n\n")
        
        # Detailed per dataset
        for dataset_name, dataset_results in results.items():
            f.write(f"## {dataset_name.upper()}\n\n")
            
            # Best performer
            best_recall = 0
            best_exp = None
            for exp_name, exp_data in dataset_results.items():
                if 10 in exp_data.get('N_results', {}):
                    recall = exp_data['N_results'][10]['metrics'].get('recall', 0)
                    if recall > best_recall:
                        best_recall = recall
                        best_exp = exp_name
            
            if best_exp:
                f.write(f"**Best Configuration:** {best_exp} (Recall@10: {best_recall*100:.2f}%)\n\n")
            
            # Results table
            f.write("### Results by Experiment\n\n")
            f.write("| Experiment | k | m | epochs | T | Recall@10 | AF | QPS |\n")
            f.write("|------------|---|---|--------|---|-----------|----|----- |\n")
            
            for exp_name, exp_data in dataset_results.items():
                params = exp_data['params']
                
                if 10 in exp_data.get('N_results', {}):
                    metrics = exp_data['N_results'][10]['metrics']
                    f.write(f"| {exp_name} | {params['knn']} | {params['m']} | {params['epochs']} | {params['T']} | ")
                    f.write(f"{metrics.get('recall', 0)*100:.2f}% | {metrics.get('avg_af', 0):.4f} | {metrics.get('qps', 0):.2f} |\n")
            
            f.write("\n")
            
            # Detailed results per N
            f.write("### Detailed Results (All N values)\n\n")
            
            for exp_name, exp_data in dataset_results.items():
                f.write(f"#### {exp_name}\n\n")
                f.write(f"*{params['description']}*\n\n")
                
                f.write("| N | Recall@N | Avg AF | QPS | t_approx (ms) |\n")
                f.write("|---|----------|--------|-----|---------------|\n")
                
                for N in sorted(exp_data.get('N_results', {}).keys()):
                    metrics = exp_data['N_results'][N]['metrics']
                    f.write(f"| {N} | {metrics.get('recall', 0)*100:.2f}% | {metrics.get('avg_af', 0):.4f} | ")
                    f.write(f"{metrics.get('qps', 0):.2f} | {metrics.get('t_approx', 0)*1000:.2f} |\n")
                
                f.write("\n")
            
            f.write("---\n\n")
    
    print(f"📄 Report generated: {report_file}")
    return report_file


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Neural LSH experiments v2")
    parser.add_argument("--datasets", nargs="+", 
                        default=['sift', 'mnist'],
                        choices=['sift', 'sift_full', 'mnist'],
                        help="Datasets to test (sift uses sift_learn for training)")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Specific experiments to run (default: all)")
    parser.add_argument("--N", nargs="+", type=int, default=[1, 5, 10],
                        help="N values for k-NN search")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with only baseline + optimal")
    parser.add_argument("--improved", action="store_true",
                        help="Use improved build script with scheduler")
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        experiments = ['baseline', 'optimal']
        N_values = [10]
    else:
        experiments = args.experiments
        N_values = args.N
    
    # Create output directories
    os.makedirs("indexes", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run experiments
    results, results_file = run_all_experiments(
        datasets=args.datasets,
        experiments=experiments,
        N_values=N_values,
        use_improved=args.improved
    )
    
    if results:
        # Generate report
        report_file = generate_markdown_report(results, results_file)
        
        print("\n✨ Experiments completed successfully!")
        print(f"📊 Results: {results_file}")
        print(f"📄 Report: {report_file}")