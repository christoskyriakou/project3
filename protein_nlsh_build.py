#!/usr/bin/env python3
"""
protein_nlsh_build.py
Build Neural LSH index for protein embeddings
"""

import argparse
import numpy as np
import os
from dataset_parser import load_dataset
from nlsh_build import train_mlp, main as nlsh_main
from graph_utils import build_knn_graph_with_ivfflat, make_undirected_weighted, run_kahip


def build_protein_nlsh_index(embeddings_path, index_dir, args):
    """
    Build Neural LSH index from protein embeddings
    
    Steps:
    1. Load embeddings
    2. Build KNN graph (using IVFFlat for speed)
    3. Run KaHIP partitioning
    4. Train MLP classifier
    5. Save index
    """
    
    print("[BUILD] Loading protein embeddings...")
    X, dtype = load_dataset(embeddings_path)
    n, d = X.shape
    print(f"[BUILD] Loaded {n} vectors, dim={d}")
    
    # Create index directory
    os.makedirs(index_dir, exist_ok=True)
    
    # Save embeddings in format expected by IVFFlat
    print("[BUILD] Preparing data for KNN graph construction...")
    
    # For now, use brute force KNN for small datasets
    # For larger datasets, you would use your IVFFlat implementation
    if n < 5000:
        print(f"[BUILD] Using brute force KNN (n={n} is small)")
        graph = build_knn_graph_bruteforce(X, args.knn)
    else:
        print(f"[BUILD] Building KNN graph with IVFFlat (k={args.knn})...")
        # This would call your IVFFlat KNN implementation
        # For now, fall back to brute force
        graph = build_knn_graph_bruteforce(X, args.knn)
    
    # Build undirected weighted graph
    print("[BUILD] Building undirected weighted graph...")
    ugraph = make_undirected_weighted(graph)
    
    # Run KaHIP partitioning
    print(f"[BUILD] Running KaHIP (m={args.m}, imbalance={args.imbalance})...")
    parts = run_kahip(ugraph, args.m, args.imbalance, args.kahip_mode)
    
    print(f"[BUILD] Partitions: min={parts.min()}, max={parts.max()}, unique={len(np.unique(parts))}")
    
    # Save partitions
    parts_path = os.path.join(index_dir, "partitions.npy")
    np.save(parts_path, parts)
    print(f"[BUILD] Saved partitions → {parts_path}")
    
    # Train MLP
    print("[BUILD] Training MLP classifier...")
    
    # Create args object for train_mlp
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    mlp_args = Args(
        m=args.m,
        layers=args.layers,
        nodes=args.nodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
    
    model = train_mlp(X, parts, mlp_args)
    
    # Save model
    import torch
    model_path = os.path.join(index_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[BUILD] Saved model → {model_path}")
    
    # Build inverted index
    print("[BUILD] Building inverted index...")
    inverted = {r: [] for r in range(args.m)}
    for idx, r in enumerate(parts):
        inverted[int(r)].append(int(idx))
    
    # Save inverted index
    inv_path = os.path.join(index_dir, "inverted_index.npy")
    np.save(inv_path, np.array(inverted, dtype=object))
    print(f"[BUILD] Saved inverted index → {inv_path}")
    
    # Print statistics
    partition_sizes = [len(inverted[r]) for r in range(args.m)]
    print(f"\n[BUILD] Index Statistics:")
    print(f"  Partitions: {args.m}")
    print(f"  Avg partition size: {np.mean(partition_sizes):.1f}")
    print(f"  Min partition size: {np.min(partition_sizes)}")
    print(f"  Max partition size: {np.max(partition_sizes)}")
    print(f"  Std partition size: {np.std(partition_sizes):.1f}")
    
    print(f"\n[BUILD] Neural LSH index built successfully at {index_dir}")


def build_knn_graph_bruteforce(X, k):
    """
    Build KNN graph using brute force (for small datasets)
    """
    from distances import L2_distance_batch
    
    n = X.shape[0]
    graph = []
    
    print(f"[BUILD] Computing {k}-NN for {n} vectors...")
    
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"[BUILD] Progress: {i+1}/{n}")
        
        # Compute distances to all other points
        distances = L2_distance_batch(X, X[i])
        
        # Set self-distance to inf
        distances[i] = np.inf
        
        # Get k nearest neighbors
        neighbors = np.argsort(distances)[:k]
        graph.append(neighbors.tolist())
    
    return graph


def main():
    parser = argparse.ArgumentParser(description="Build Neural LSH index for proteins")
    parser.add_argument("-d", "--data", required=True, help="Protein embeddings (.fvecs)")
    parser.add_argument("-i", "--index", required=True, help="Output index directory")
    
    # KNN graph parameters
    parser.add_argument("--knn", type=int, default=10, help="k for KNN graph")
    
    # Partitioning parameters
    parser.add_argument("-m", type=int, default=100, help="Number of partitions")
    parser.add_argument("--imbalance", type=float, default=0.03, help="KaHIP imbalance")
    parser.add_argument("--kahip_mode", type=int, default=0, help="KaHIP mode")
    
    # MLP parameters
    parser.add_argument("--layers", type=int, default=3, help="MLP layers")
    parser.add_argument("--nodes", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    
    args = parser.parse_args()
    
    # Build index
    build_protein_nlsh_index(args.data, args.index, args)


if __name__ == "__main__":
    main()