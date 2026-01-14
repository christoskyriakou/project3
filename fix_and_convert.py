#!/usr/bin/env python3
import argparse
import os
import struct
import numpy as np

def load_auto(path, dim):
    """Φορτώνει αυτόματα είτε NPY είτε Raw"""
    print(f"[LOAD] Trying to load {path}...")
    try:
        # Πρώτη προσπάθεια: Ως NumPy array
        X = np.load(path)
        print("[LOAD] Success: Detected NumPy format.")
    except Exception:
        # Δεύτερη προσπάθεια: Ως raw binary
        print("[LOAD] NumPy load failed. Trying raw float32...")
        X = np.fromfile(path, dtype=np.float32)
    
    # Έλεγχοι διαστάσεων και καθαρισμός header αν χρειαστεί
    if X.ndim == 1:
        if X.size % dim != 0:
            # Αν διαβάστηκε raw αλλά έχει header 128 bytes (32 floats)
            remainder = X.size % dim
            if remainder == 32: 
                print("[WARN] Trimming NPY header manually...")
                X = X[32:] 
        
        n = X.size // dim
        X = X[:n*dim].reshape(n, dim)
        
    return X.astype(np.float32)

def write_float32(path, X):
    """
    Γράφει τα vectors ως FLOAT32 (4 bytes).
    Αυτό είναι το σωστό format για τον C κώδικα που κάνει fread float.
    """
    # ΜΕΤΑΤΡΟΠΗ ΣΕ FLOAT32
    X32 = X.astype(np.float32) 
    n, d = X32.shape
    
    print(f"[WRITE] Writing {n} vectors (dim={d}) as FLOAT32 to {path}...")
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", d))
            f.write(X32[i].tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input file (protein_vectors.dat)")
    parser.add_argument("-o", "--output", required=True, help="Output file (.fvecs)")
    parser.add_argument("--dim", type=int, default=320)
    args = parser.parse_args()

    # 1. Load
    X = load_auto(args.input, args.dim)
    print(f"[INFO] Loaded shape: {X.shape}")
    
    # 2. Normalize (Απαραίτητο για το LSH/Cosine)
    print("[PROC] Normalizing vectors...")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X = X / norms
    
    # 3. Write
    write_float32(args.output, X)
    print("[DONE] Success.")

if __name__ == "__main__":
    main()