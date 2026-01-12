import argparse
import numpy as np
import torch
import time
from models import MLP
from dataset_parser import load_dataset
from distances import L2_distance_batch
import os


def load_index(index_path, d, m, layers, nodes):
    """
    Φορτώνει το εκπαιδευμένο μοντέλο και το inverted index
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = os.path.join(index_path, "model.pth")
    model = MLP(d_in=d, num_classes=m, layers=layers, hidden=nodes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[SEARCH] Loaded model from {model_path}")
    
    # Load inverted index
    inv_path = os.path.join(index_path, "inverted_index.npy")
    inverted = np.load(inv_path, allow_pickle=True).item()
    print(f"[SEARCH] Loaded inverted index from {inv_path}")
    
    return model, inverted, device


def multi_probe_search(model, inverted, query, T, device):
    """
    Εκτελεί multi-probe search για ένα query
    
    Returns:
        candidate_indices: λίστα με τα indices των υποψηφίων σημείων
    """
    q_tensor = torch.from_numpy(query).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(q_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # Βρες τα T bins με τις μεγαλύτερες πιθανότητες
    top_bins = np.argsort(probs)[-T:][::-1]
    
    # Συλλογή υποψηφίων από τα T bins
    candidates = []
    for bin_id in top_bins:
        if bin_id in inverted:
            candidates.extend(inverted[bin_id])
    
    return list(set(candidates))  # Remove duplicates


def knn_search(X, query, candidate_indices, N):
    """
    Εκτελεί ακριβή k-NN search στους υποψηφίους
    
    Returns:
        neighbors: λίστα με (index, distance) για τους N πλησιέστερους
    """
    if len(candidate_indices) == 0:
        return []
    
    # Υπολόγισε αποστάσεις μόνο για τους υποψηφίους
    candidates_data = X[candidate_indices]
    distances = L2_distance_batch(candidates_data, query)
    
    # Ταξινόμηση και επιλογή top-N
    sorted_idx = np.argsort(distances)[:N]
    
    neighbors = [(candidate_indices[i], distances[i]) for i in sorted_idx]
    return neighbors


def range_search(X, query, candidate_indices, R):
    """
    Εκτελεί range search στους υποψηφίους
    
    Returns:
        neighbors: λίστα με indices εντός ακτίνας R
    """
    if len(candidate_indices) == 0:
        return []
    
    candidates_data = X[candidate_indices]
    distances = L2_distance_batch(candidates_data, query)
    
    # Φιλτράρισμα με βάση την ακτίνα
    within_range = [candidate_indices[i] for i in range(len(distances)) if distances[i] <= R]
    return within_range


# BRUTE FORCE
def true_knn_search(X, query, N):
    """
    Εξαντλητική αναζήτηση σε ολόκληρο το dataset
    """
    distances = L2_distance_batch(X, query)
    sorted_idx = np.argsort(distances)[:N]
    neighbors = [(i, distances[i]) for i in sorted_idx]
    return neighbors


def compute_metrics(approx_neighbors, true_neighbors, range_neighbors, t_approx, t_true):
    """
    Υπολογίζει AF, Recall@N
    """
    # Average AF (Approximation Factor)
    af_sum = 0.0
    count = 0
    
    for (idx_a, dist_a), (idx_t, dist_t) in zip(approx_neighbors, true_neighbors):
        if dist_t > 0:
            af_sum += dist_a / dist_t
            count += 1
    
    avg_af = af_sum / count if count > 0 else 1.0
    
    # Recall@N
    approx_set = set([idx for idx, _ in approx_neighbors])
    true_set = set([idx for idx, _ in true_neighbors])
    recall = len(approx_set & true_set) / len(true_set) if len(true_set) > 0 else 0.0
    
    return avg_af, recall


def main():
    parser = argparse.ArgumentParser(description="Neural LSH Search")
    parser.add_argument("-d", "--data", required=True, help="Input dataset file")
    parser.add_argument("-q", "--query", required=True, help="Query file")
    parser.add_argument("-i", "--index", required=True, help="Index directory")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("-type", "--type", required=True, choices=["sift", "mnist"])
    parser.add_argument("-N", type=int, default=1, help="Number of nearest neighbors")
    parser.add_argument("-R", type=float, default=None, help="Range search radius")
    parser.add_argument("-T", type=int, default=5, help="Number of bins to probe")
    parser.add_argument("-range", type=str, default="true", choices=["true", "false"])
    
    # Parameters για το μοντέλο (πρέπει να είναι ίδιες με το build)
    parser.add_argument("-m", type=int, help="Number of partitions")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=64)
    
    args = parser.parse_args()
    
    # Default radius values
    if args.R is None:
        args.R = 2000 if args.type == "mnist" else 2800
    
    do_range = args.range.lower() == "true"
    
    # Load dataset and queries
    print("[SEARCH] Loading dataset...")
    X, _ = load_dataset(args.data)
    n, d = X.shape
    print(f"[SEARCH] Dataset: {n} vectors, dim={d}")
    
    print("[SEARCH] Loading queries...")
    Q, _ = load_dataset(args.query)
    print(f"[SEARCH] Queries: {Q.shape[0]} vectors")
    
    # Load index
    print("[SEARCH] Loading index...")

    # Auto-detect m if not provided
    if args.m is None:
        inv_path = os.path.join(args.index, "inverted_index.npy")
        if os.path.exists(inv_path):
            temp_inv = np.load(inv_path, allow_pickle=True).item()
            args.m = len(temp_inv)
            print(f"[SEARCH] Auto-detected m={args.m} from index")
        else:
            print(f"[SEARCH] ERROR: Could not auto-detect m and --m not specified")
            return

    model, inverted, device = load_index(args.index, d, args.m, args.layers, args.nodes)
    
    # Process queries
    results = []
    total_af = 0.0
    total_recall = 0.0
    total_t_approx = 0.0
    total_t_true = 0.0
    
    print(f"[SEARCH] Processing {Q.shape[0]} queries...")
    
    for q_idx, query in enumerate(Q):
        # Approximate search
        t_start = time.time()
        candidate_indices = multi_probe_search(model, inverted, query, args.T, device)
        approx_neighbors = knn_search(X, query, candidate_indices, args.N)
        t_approx = time.time() - t_start
        
        # Range search (if enabled)
        range_neighbors = []
        if do_range:
            range_neighbors = range_search(X, query, candidate_indices, args.R)
        
        # True search (brute force)
        t_start = time.time()
        true_neighbors = true_knn_search(X, query, args.N)
        t_true = time.time() - t_start
        
        # Compute metrics
        af, recall = compute_metrics(approx_neighbors, true_neighbors, range_neighbors, t_approx, t_true)
        
        total_af += af
        total_recall += recall
        total_t_approx += t_approx
        total_t_true += t_true
        
        # Store results
        results.append({
            'query_id': q_idx,
            'approx_neighbors': approx_neighbors,
            'true_neighbors': true_neighbors,
            'range_neighbors': range_neighbors,
            'af': af,
            'recall': recall
        })
        
        if (q_idx + 1) % 10 == 0:
            print(f"[SEARCH] Processed {q_idx + 1}/{Q.shape[0]} queries")
    
    # Compute averages
    num_queries = Q.shape[0]
    avg_af = total_af / num_queries
    avg_recall = total_recall / num_queries
    qps = num_queries / total_t_approx
    avg_t_approx = total_t_approx / num_queries
    avg_t_true = total_t_true / num_queries
    
    print(f"\n[SEARCH] === RESULTS ===")
    print(f"[SEARCH] Average AF: {avg_af:.4f}")
    print(f"[SEARCH] Recall@{args.N}: {avg_recall:.4f}")
    print(f"[SEARCH] QPS: {qps:.2f}")
    print(f"[SEARCH] Avg t_approx: {avg_t_approx:.6f}s")
    print(f"[SEARCH] Avg t_true: {avg_t_true:.6f}s")
    
    # Write output file
    print(f"[SEARCH] Writing results to {args.output}...")
    
    with open(args.output, 'w') as f:
        f.write("Neural LSH\n")
        
        for res in results:
            f.write(f"Query: {res['query_id']}\n")
            
            for i, (idx, dist_a) in enumerate(res['approx_neighbors'], 1):
                _, dist_t = res['true_neighbors'][i-1]
                f.write(f"Nearest neighbor-{i}: {idx}\n")
                f.write(f"distanceApproximate: {dist_a:.6f}\n")
                f.write(f"distanceTrue: {dist_t:.6f}\n")
            
            if do_range:
                f.write("R-near neighbors:\n")
                for idx in res['range_neighbors']:
                    f.write(f"{idx}\n")
        
        f.write(f"Average AF: {avg_af:.6f}\n")
        f.write(f"Recall@{args.N}: {avg_recall:.6f}\n")
        f.write(f"QPS: {qps:.2f}\n")
        f.write(f"tApproximateAverage: {avg_t_approx:.6f}\n")
        f.write(f"tTrueAverage: {avg_t_true:.6f}\n")
    
    print(f"[SEARCH] Output written successfully to {args.output}")
    print("[SEARCH] Search completed!")


if __name__ == "__main__":
    main()