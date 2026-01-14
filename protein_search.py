#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import time
import struct
import tempfile
import numpy as np
import torch
import esm
from Bio import SeqIO

# --- ΡΥΘΜΙΣΕΙΣ ΕΚΤΕΛΕΣΙΜΩΝ ---
BINARY_LSH      = "./search"        
BINARY_HYPERCUBE= "./hypercube" 
BINARY_IVF_FLAT = "./ivf_flat"
BINARY_IVF_PQ   = "./ivf_pq"

ID_MAPPING_FILE = "protein_vectors_ids.txt"

def load_fvecs(filename):
    if not os.path.exists(filename):
        print(f"[ERR] File {filename} not found.")
        sys.exit(1)
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

def write_fvecs(filename, data):
    data = data.astype(np.float32) 
    n, d = data.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(data[i].tobytes())

def l2_normalize(x):
    norm = np.linalg.norm(x)
    if norm > 1e-12: return x / norm
    return x

def load_id_mapping(filepath):
    print(f"[INFO] Loading IDs from {filepath}...")
    mapping = {}
    if not os.path.exists(filepath): return mapping
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split() 
            if len(parts) >= 2:
                try: mapping[int(parts[0])] = parts[1]
                except: continue
    return mapping

def embed_queries(fasta_path):
    print(f"[INIT] Embedding queries...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    queries = []
    ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        ids.append(record.id)
        queries.append((record.id, str(record.seq)[:1022])) 
    embeddings = []
    for i, (qid, seq) in enumerate(queries):
        batch_labels, batch_strs, batch_tokens = batch_converter([(qid, seq)])
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        token_embeddings = results["representations"][6]
        seq_embed = token_embeddings[0, 1 : len(seq) + 1].mean(0).cpu().numpy()
        seq_embed = l2_normalize(seq_embed)
        embeddings.append(seq_embed)
    return np.array(embeddings, dtype=np.float32), ids

def setup_blast_db(fasta_path):
    db_name = fasta_path 
    if not os.path.exists(fasta_path + ".phr"):
        subprocess.run(["makeblastdb", "-in", fasta_path, "-dbtype", "prot", "-out", db_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return db_name

def run_blast(query_seq, query_id, db_name, top_n=50):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as q_file:
        q_file.write(f">{query_id}\n{query_seq}\n")
        q_file.flush()
        cmd = ["blastp", "-query", q_file.name, "-db", db_name, "-outfmt", "6 sseqid pident evalue", "-max_target_seqs", str(top_n)]
        try:
            start = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            dur = time.time() - start
        except: return [], 0.0
    hits = []
    for line in res.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) >= 3:
            hits.append({'id': parts[0], 'identity': float(parts[1])})
    return hits[:top_n], dur

# --- NEO PARSER ΓΙΑ BATCH RESULTS ---
def parse_batch_c_output(output_file, id_mapping, num_queries):
    # Δομή: { query_index: [ {id, dist, index}, ... ] }
    all_results = {i: [] for i in range(num_queries)}
    
    current_query_idx = -1
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # Ανιχνεύουμε αλλαγή query: "Query: 5"
            if line.startswith("Query:"):
                try:
                    current_query_idx = int(line.split(':')[1].strip())
                except: pass
                continue
            
            # Ανιχνεύουμε γείτονα
            if line.startswith("Nearest neighbor"):
                if current_query_idx == -1: continue # Δεν έχουμε βρει query header ακόμα
                
                # Καθαρισμός ID
                val_str = line.split(':')[1].strip().replace("image_id_", "")
                try: 
                    idx = int(val_str)
                    
                    # Διαβάζουμε την επόμενη γραμμή για απόσταση (προαιρετικό Hack, ή υποθέτουμε ότι είναι στην επόμενη)
                    # Εδώ κάνουμε μια παραδοχή: Το C πρόγραμμα τυπώνει:
                    # Nearest neighbor-1: 123
                    # distanceApproximate: 0.45
                    # Θα αποθηκεύσουμε το index και θα περιμένουμε την απόσταση
                    
                    pid = id_mapping.get(idx, f"Unknown_{idx}")
                    all_results[current_query_idx].append({'index': idx, 'id': pid, 'dist': -1.0})
                    
                except: continue

            elif line.startswith("distanceApproximate"):
                if current_query_idx != -1 and all_results[current_query_idx]:
                    # Βάζουμε την απόσταση στον τελευταίο γείτονα που προσθέσαμε
                    dist = float(line.split(':')[1].strip())
                    all_results[current_query_idx][-1]['dist'] = dist

    except Exception as e:
        print(f"[WARN] Error parsing output: {e}")
        
    return all_results

# --- BATCH RUN FUNCTION ---
def run_c_search_batch(method, db_file, all_queries_file, N, params, id_mapping, num_queries):
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    out_tmp.close()
    
    cmd = []
    if method == "lsh": 
        cmd = [BINARY_LSH, "-d", db_file, "-q", all_queries_file, "-o", out_tmp.name, "-N", str(N), "-type", "sift"]
        cmd += ["-lsh", "-k", str(params['lsh_k']), "-L", str(params['lsh_L']), "-w", str(params['lsh_w'])]
    elif method == "hypercube": 
        cmd = [BINARY_HYPERCUBE, "-d", db_file, "-q", all_queries_file, "-o", out_tmp.name, "-N", str(N), "-type", "sift"]
        cmd += ["-kproj", str(params['hc_k']), "-M", str(params['hc_M']), "-probes", str(params['hc_probes']), "-w", str(params['hc_w'])]
    elif method == "ivf": 
        cmd = [BINARY_IVF_FLAT, "-d", db_file, "-q", all_queries_file, "-o", out_tmp.name, "-N", str(N), "-type", "sift"]
        cmd += ["-ivfflat", "-kclusters", str(params['ivf_k']), "-nprobe", str(params['ivf_probe'])]
    elif method == "ivfpq":
        cmd = [BINARY_IVF_PQ, "-d", db_file, "-q", all_queries_file, "-o", out_tmp.name, "-N", str(N), "-type", "sift"]
        cmd += ["-ivfpq", "-kclusters", str(params['ivf_k']), "-nprobe", str(params['ivf_probe']), "-M", "16", "-nbits", "8"]

    print(f"   [BATCH] Running {method} on all queries...")
    start = time.time()
    try:
        # 10 λεπτά timeout για ΟΛΑ τα queries μαζί
        subprocess.run(cmd, timeout=600) 
        dur = time.time() - start
        
        # Parse output for ALL queries
        batch_results = parse_batch_c_output(out_tmp.name, id_mapping, num_queries)
        
    except subprocess.TimeoutExpired:
        print(f"[WARN] Timeout for {method} (exceeded 600s)")
        dur = 600.0
        batch_results = {i: [] for i in range(num_queries)}
    
    if os.path.exists(out_tmp.name): os.remove(out_tmp.name)
    
    # Επιστρέφουμε αποτελέσματα και ΜΕΣΟ χρόνο ανά query
    avg_time = dur / num_queries if num_queries > 0 else 0
    return batch_results, avg_time

def check_match(blast_ids, found_id):
    if found_id in blast_ids: return True
    for bid in blast_ids:
        if found_id in bid or bid in found_id: return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--database", required=True)
    parser.add_argument("-q", "--queries", required=True)
    parser.add_argument("-s", "--swissprot", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-method", default="all") 
    parser.add_argument("-N", type=int, default=50)
    
    # Παράμετροι
    parser.add_argument("--lsh_k", type=int, default=2) 
    parser.add_argument("--lsh_L", type=int, default=64)
    parser.add_argument("--lsh_w", type=float, default=1.4)
    parser.add_argument("--hc_k", type=int, default=9) 
    parser.add_argument("--hc_M", type=int, default=5000)
    parser.add_argument("--hc_probes", type=int, default=50) 
    parser.add_argument("--hc_w", type=float, default=1.2)
    parser.add_argument("--ivf_k", type=int, default=10)
    parser.add_argument("--ivf_probe", type=int, default=2)
    
    args = parser.parse_args()
    id_mapping = load_id_mapping(ID_MAPPING_FILE)
    
    db_vectors = load_fvecs(args.database)
    query_vecs, query_ids = embed_queries(args.queries)
    query_seqs = {r.id: str(r.seq) for r in SeqIO.parse(args.queries, "fasta")}
    blast_db = setup_blast_db(args.swissprot)
    
    # --- 1. PREPARE BATCH QUERY FILE ---
    print(f"[INIT] Creating batch query file for {len(query_vecs)} queries...")
    batch_query_file = "all_queries_batch.fvecs"
    write_fvecs(batch_query_file, query_vecs)
    
    # --- 2. RUN METHODS (BATCH) ---
    methods = ["lsh", "hypercube", "ivf", "ivfpq"] if args.method == "all" else [args.method]
    method_results = {} # Store results: method -> { query_idx: [results] }
    method_times = {}   # Store times: method -> avg_time_per_query
    
    print("\n[START] Executing C Binaries in BATCH mode...")
    for m in methods:
        res, t = run_c_search_batch(m, args.database, batch_query_file, args.N, vars(args), id_mapping, len(query_vecs))
        method_results[m] = res
        method_times[m] = t
        print(f"   -> {m} finished. Avg time/query: {t:.4f}s")

    # --- 3. RUN BLAST & COMPARE ---
    f_out = open(args.output, 'w')
    print(f"\n[START] Running BLAST & Generating Report...")
    
    for i, qid in enumerate(query_ids):
        print(f" -> Comparing Query {i+1}/{len(query_ids)}: {qid}")
        f_out.write(f"Query: {qid}\n")
        
        # BLAST (Per query, as it's fast enough and hard to batch via biopython simply)
        blast_hits, blast_t = run_blast(query_seqs[qid], qid, blast_db, args.N)
        blast_ids = set([h['id'] for h in blast_hits])
        
        f_out.write(f"BLAST (Benchmark) | Time: {blast_t:.3f} | Recall: 1.0\n")
        f_out.write(f"Top 10 BLAST Hits:\n")
        for hit in blast_hits[:10]:
            f_out.write(f"{hit['id']} (Identity: {hit['identity']:.1f}%)\n")
        f_out.write("-" * 40 + "\n")
        
        # RESULTS FROM C METHODS
        for m in methods:
            # Get pre-calculated results for this query index
            res = method_results[m].get(i, [])
            t = method_times[m]
            
            matches = 0
            for r in res:
                if check_match(blast_ids, r['id']): matches += 1
            recall = matches / len(blast_ids) if blast_ids else 0.0
            
            f_out.write(f"{m} | Time (avg): {t:.3f} | Recall vs BLAST: {recall:.3f}\n")
            f_out.write(f"Top 10 {m}:\n")
            for r in res[:10]:
                note = " [MATCH]" if check_match(blast_ids, r['id']) else ""
                d_val = r.get('dist', -1.0)
                f_out.write(f"{r['id']} (d={d_val:.4f}){note}\n")
        f_out.write("\n")
        f_out.flush()
        
    f_out.close()
    if os.path.exists(batch_query_file): os.remove(batch_query_file)
    print("[DONE]")

if __name__ == "__main__":
    main()