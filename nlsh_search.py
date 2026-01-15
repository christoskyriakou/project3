#!/usr/bin/env python3
import argparse
import numpy as np
import os
import struct
import torch
import torch.nn as nn
import sys

def load_fvecs(filename):
    if not os.path.exists(filename):
        sys.stderr.write(f"[ERR] File {filename} not found.\n")
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

class NeuralLSHModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(NeuralLSHModel, self).__init__()
        layers = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--database", required=True)
    parser.add_argument("-q", "--queries", required=True)
    parser.add_argument("-i", "--model", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-N", type=int, default=50)
    parser.add_argument("-type", default="sift") 
    args = parser.parse_args()

    X_db = load_fvecs(args.database)
    X_q = load_fvecs(args.queries)
    n_db, d = X_db.shape
    n_q, _ = X_q.shape

    # Load Index safely
    paths = ["protein_index/inverted_index.npy", "inverted_index.npy"]
    inv_path = next((p for p in paths if os.path.exists(p)), None)
    if not inv_path: return

    loaded = np.load(inv_path, allow_pickle=True)
    inverted_index = loaded.item() if loaded.ndim == 0 else loaded
    # Convert list/array to dict
    if not isinstance(inverted_index, dict):
        inverted_index = {i: v for i, v in enumerate(inverted_index)}

    # Load Model
    model = NeuralLSHModel(d, 128, len(inverted_index), num_layers=2)
    try:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
    except: pass
    model.eval()

    f_out = open(args.output, 'w')
    q_tensor = torch.from_numpy(X_q)
    
    with torch.no_grad():
        logits = model(q_tensor)
        # Check Top-5 partitions
        _, top_parts = torch.topk(logits, 5, dim=1)
        top_parts = top_parts.numpy()

    for i in range(n_q):
        query_vec = X_q[i]
        candidates = set()
        for p_idx in top_parts[i]:
            bucket = inverted_index.get(int(p_idx))
            if bucket is not None:
                if isinstance(bucket, (list, np.ndarray)): candidates.update(bucket)
        
        candidates = list(candidates)
        f_out.write(f"Query: {i}\n")
        
        if candidates:
            cand_vecs = X_db[candidates]
            dists = np.linalg.norm(cand_vecs - query_vec, axis=1)
            sorted_idx = np.argsort(dists)[:args.N]
            for local_idx in sorted_idx:
                f_out.write(f"Nearest neighbor: {candidates[local_idx]}\n")
                f_out.write(f"distanceTrue: {dists[local_idx]}\n")
    f_out.close()

if __name__ == "__main__":
    main()