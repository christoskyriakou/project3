#!/usr/bin/env python3
"""
protein_nlsh_build.py - FIXED VERSION
"""
import argparse
import numpy as np
import os
import struct
import torch
# Υποθέτουμε ότι αυτά τα imports υπάρχουν στο project σου
from nlsh_build import train_mlp
from graph_utils import make_undirected_weighted, run_kahip

def load_fvecs(filename):
    if not os.path.exists(filename):
        print(f"[ERR] File {filename} not found.")
        return np.zeros((0, 0))
    with open(filename, 'rb') as f:
        d_bytes = f.read(4)
        if not d_bytes: return np.zeros((0, 0))
        d = struct.unpack('i', d_bytes)[0]
        f.seek(0, 2)
        n = f.tell() // (4 + d * 4) 
        f.seek(0)
        X = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            f.read(4) 
            X[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return X

def build_knn_graph_bruteforce(X, k):
    """ Brute force KNN for vectors """
    n, d = X.shape
    graph = []
    print(f"[BUILD] Computing {k}-NN for {n} vectors (Brute Force)...")
    # Χωρίζουμε σε blocks για να μην γεμίσει η μνήμη
    block_size = 1000
    for i in range(0, n, block_size):
        end = min(i + block_size, n)
        batch = X[i:end]
        # Dot product αφού είναι normalized = Cosine Similarity
        # Μεγαλύτερο dot = μικρότερη απόσταση
        sims = np.dot(batch, X.T)
        for j in range(len(batch)):
            sims[j, i+j] = -np.inf # Ignore self
            # Παίρνουμε τα k μεγαλύτερα (top-k neighbors)
            indices = np.argpartition(sims[j], -k)[-k:]
            graph.append(indices.tolist())
        if (i // block_size) % 5 == 0:
            print(f"   Processed {end}/{n}...")
    return graph

def build_protein_nlsh_index(embeddings_path, index_dir, args):
    print(f"[BUILD] Loading protein embeddings from {embeddings_path}...")
    X = load_fvecs(embeddings_path)
    n, d = X.shape
    print(f"[BUILD] Loaded {n} vectors, dim={d}")

    # --- [FIX] NORMALIZATION ---
    print("[FIX] Normalizing vectors (L2) for training...")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-10)
    print("[FIX] Normalization Done.")
    # ---------------------------

    os.makedirs(index_dir, exist_ok=True)
    
    # KNN Graph
    graph = build_knn_graph_bruteforce(X, args.knn)
    ugraph = make_undirected_weighted(graph)
    
    # KaHIP
    print(f"[BUILD] Running KaHIP (m={args.m})...")
    parts = run_kahip(ugraph, args.m, args.imbalance, args.kahip_mode)
    
    # Save partitions
    np.save(os.path.join(index_dir, "partitions.npy"), parts)
    
    # Train MLP
    print("[BUILD] Training MLP classifier...")
    class Args:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    
    mlp_args = Args(m=args.m, layers=args.layers, nodes=args.nodes, 
                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    
    model = train_mlp(X, parts, mlp_args)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(index_dir, "model.pth"))
    print(f"[BUILD] Saved model.")
    
    # Inverted Index
    print("[BUILD] Building inverted index...")
    inverted = {r: [] for r in range(args.m)}
    for idx, r in enumerate(parts):
        inverted[int(r)].append(int(idx))
    
    np.save(os.path.join(index_dir, "inverted_index.npy"), np.array(inverted, dtype=object))
    print(f"[BUILD] Index built successfully at {index_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("-m", type=int, default=100)
    parser.add_argument("--imbalance", type=float, default=0.03)
    parser.add_argument("--kahip_mode", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    build_protein_nlsh_index(args.data, args.index, args)

if __name__ == "__main__":
    main()