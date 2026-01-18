#!/usr/bin/env python3
"""
Protein Sequence Search Benchmark Script (Final Version)
------------------------------------------------------
Workflow:
1. Embeds query sequences using ESM-2 (Hugging Face Transformers).
2. Runs BLAST to establish ground truth.
3. Normalizes Database Vectors (Critical for Cosine Similarity via L2).
4. Runs geometric searches (C binaries) on normalized vectors.
5. Generates the comparison report.
"""

import argparse
import os
import sys
import subprocess
import time
import struct
import tempfile
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from difflib import SequenceMatcher

BINARY_LSH       = "./search"        
BINARY_HYPERCUBE = "./hypercube" 
BINARY_IVF_FLAT  = "./ivf_flat"
BINARY_IVF_PQ    = "./ivf_pq"

#         Embedding Utility Functions

def mean_pool(token_embeddings, attention_mask):
    """Mean pooling: average embeddings across sequence length."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def l2_normalize(x):
    """Normalizes a vector to unit length (L2 norm = 1)."""
    norm = np.linalg.norm(x)
    if norm > 1e-12: return x / norm
    return x

#         Vector I/O Utility Functions

def load_fvecs(filename):
    """Reads vectors from a binary .fvecs file."""
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
    """Writes a numpy array to .fvecs binary format."""
    data = data.astype(np.float32) 
    n, d = data.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('i', d))
            f.write(data[i].tobytes())

def load_id_mapping(filepath):
    """Loads a mapping file (Index -> Protein ID)."""
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

#       Bioinformatics / BLAST Utilities

def load_seq_db(fasta_path):
    """Parses FASTA to create dictionaries of sequences and descriptions."""
    print(f"[INFO] Loading Sequences from {fasta_path}...")
    seq_db = {}
    desc_db = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_db[record.id] = str(record.seq)
        
        full_desc = record.description
        # Clean up SwissProt descriptions
        if "Full=" in full_desc:
            try: clean_desc = full_desc.split("Full=")[1].split(";")[0]
            except: clean_desc = full_desc.split(None, 1)[-1]
        else:
            parts = full_desc.split(None, 1)
            clean_desc = parts[1] if len(parts) > 1 else record.id

        if "OS=" in clean_desc:
            clean_desc = clean_desc.split("OS=")[0].strip()
            
        desc_db[record.id] = clean_desc
    return seq_db, desc_db

def calc_identity(seq1, seq2):
    """Calculates sequence identity percentage."""
    if not seq1 or not seq2: return 0.0
    return SequenceMatcher(None, seq1, seq2).ratio() * 100.0

def embed_queries(fasta_path, batch_size=8):
    """Embeds queries on-the-fly and L2-normalizes them."""
    print(f"[INIT] Embedding queries from {fasta_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model_name = "facebook/esm2_t6_8M_UR50D"
        print(f"[INIT] Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"[ERR] Failed to load model: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()
    
    queries = []
    ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        ids.append(record.id)
        queries.append(str(record.seq)[:1022]) # Truncate to avoid OOM
    
    print(f"[INIT] Loaded {len(queries)} query sequences")
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_seqs = queries[i:i+batch_size]
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"[INIT] Processing batch {i // batch_size + 1}/{(len(queries) + batch_size - 1) // batch_size}")
            
            encoded = tokenizer(batch_seqs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
            outputs = model(**encoded)
            batch_embeddings = mean_pool(outputs.last_hidden_state, encoded['attention_mask']).cpu().numpy()
            
            #  Normalize queries
            for j in range(len(batch_embeddings)):
                batch_embeddings[j] = l2_normalize(batch_embeddings[j])
            
            embeddings.extend(batch_embeddings)
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    return embeddings_array, ids

def setup_blast_db(fasta_path):
    """Creates a BLAST database if it doesn't exist."""
    db_name = fasta_path 
    if not os.path.exists(fasta_path + ".phr"):
        print(f"[INFO] Creating BLAST database for {fasta_path}...")
        subprocess.run(["makeblastdb", "-in", fasta_path, "-dbtype", "prot", "-out", db_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return db_name

def run_blast(query_seq, query_id, db_name, top_n=50):
    """Runs blastp for a single query."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as q_file:
        q_file.write(f">{query_id}\n{query_seq}\n")
        q_file_name = q_file.name
        
    try:
        cmd = ["blastp", "-query", q_file_name, "-db", db_name, 
               "-outfmt", "6 sseqid pident evalue", "-max_target_seqs", str(top_n)]
        start = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        dur = time.time() - start
    except Exception as e:
        print(f"[WARN] BLAST failed for {query_id}: {e}")
        return [], 0.0
    finally:
        if os.path.exists(q_file_name): os.remove(q_file_name)
        
    hits = []
    for line in res.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) >= 3:
            hits.append({'id': parts[0], 'identity': float(parts[1])})
    return hits[:top_n], dur


#  C / External Binary Handling

def parse_batch_c_output(output_file, id_mapping, num_queries):
    """Parses text output from C++ binaries."""
    all_results = {i: [] for i in range(num_queries)}
    current_query_idx = -1
    if not os.path.exists(output_file): return all_results

    try:
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Query:"):
                    try: current_query_idx = int(line.split(':')[1].strip())
                    except: pass
                    continue
                
                if line.startswith("Nearest neighbor"):
                    if current_query_idx == -1: continue
                    val_str = line.split(':')[1].strip().replace("image_id_", "")
                    try: 
                        idx = int(val_str)
                        pid = id_mapping.get(idx, f"Unknown_{idx}")
                        all_results[current_query_idx].append({'index': idx, 'id': pid, 'dist': -1.0})
                    except: continue

                elif line.startswith("distanceApproximate") or line.startswith("distanceTrue"):
                    if current_query_idx != -1 and all_results[current_query_idx]:
                        try:
                            dist = float(line.split(':')[1].strip())
                            all_results[current_query_idx][-1]['dist'] = dist
                        except: pass
    except: pass
    return all_results

def run_c_search_batch(method, db_file, all_queries_file, N, params, id_mapping, num_queries):
    """Runs the specified C++ binary."""
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    out_tmp.close()
    
    cmd = []
    # Parameters set according to Assignment Step 2
    if method == "lsh": 
        cmd = [BINARY_LSH, "-d", db_file, "-q", all_queries_file, "-o", out_tmp.name, "-N", str(N), "-type", "sift"]
        cmd += ["-lsh", "-k", str(params['lsh_k']), "-L", str(params['lsh_L']), "-w", str(params['lsh_w'])]
    elif method == "hypercube": 
        cmd = [BINARY_HYPERCUBE, "-d", db_file, "-q", all_queries_file, "-o", out_tmp.name, "-N", str(N), "-type", "sift"]
        cmd += ["-kproj", str(params['hc_k']), "-M", str(params['hc_M']), "-probes", str(params['hc_probes']), "-w", str(params['hc_w'])]
    elif method == "ivf": 
        cmd = [BINARY_IVF_FLAT, "-d", db_file, "-q", all_queries_file, "-o", out_tmp.name, "-N", str(N), "-type", "sift"]
        cmd += ["-ivfflat", "-kclusters", str(params['ivf_k']), "-nprobe", str(params['ivf_probe']), "-range", "false"]
    elif method == "ivfpq":
        cmd = [BINARY_IVF_PQ, "-d", db_file, "-q", all_queries_file, "-o", out_tmp.name, "-N", str(N), "-type", "sift"]
        cmd += ["-ivfpq", "-kclusters", str(params['ivf_k']), "-nprobe", str(params['ivf_probe']), "-M", "20", "-nbits", "8", "-range", "false"]
    elif method == "neural":    
        cmd = ["python3", "nlsh_search.py", "-d", db_file, "-q", all_queries_file, "-i", "nlsh_model.pth", 
               "-o", out_tmp.name, "-type", "sift", "-N", str(N)]
    
    print(f"   [BATCH] Running {method}...")
    start = time.time()
    try:
        subprocess.run(cmd, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        dur = time.time() - start
        batch_results = parse_batch_c_output(out_tmp.name, id_mapping, num_queries)
    except:
        dur = 600.0
        batch_results = {i: [] for i in range(num_queries)}
    
    if os.path.exists(out_tmp.name): os.remove(out_tmp.name)
    avg_time = dur / num_queries if num_queries > 0 else 0
    return batch_results, avg_time

def check_match(blast_ids, found_id):
    """Checks if found_id is in the BLAST ground truth set."""
    if found_id in blast_ids: return True
    clean_found = found_id.split('|')[-1] if '|' in found_id else found_id
    for bid in blast_ids:
        clean_blast = bid.split('|')[-1] if '|' in bid else bid
        if clean_found in clean_blast or clean_blast in clean_found: return True
    return False

#Main Execution

def main():
    parser = argparse.ArgumentParser(description="Benchmark ANN methods vs BLAST for proteins.")
    parser.add_argument("-d", "--database", required=True, help="Input vector database (.dat or .fvecs)")
    parser.add_argument("-q", "--queries", required=True, help="Input query FASTA file")
    parser.add_argument("-s", "--swissprot", default=None, help="SwissProt FASTA for BLAST (default: infer from DB name)")
    parser.add_argument("-o", "--output", required=True, help="Output report file")
    parser.add_argument("-method", default="all", help="Specific method to run (or 'all')") 
    parser.add_argument("-N", type=int, default=50, help="Recall@N threshold")
    
    # Hyperparameters
    parser.add_argument("--lsh_k", type=int, default=4) 
    parser.add_argument("--lsh_L", type=int, default=5)
    parser.add_argument("--lsh_w", type=float, default=1.4)
    parser.add_argument("--hc_k", type=int, default=14) 
    parser.add_argument("--hc_M", type=int, default=10)
    parser.add_argument("--hc_probes", type=int, default=2) 
    parser.add_argument("--hc_w", type=float, default=1.4)
    parser.add_argument("--ivf_k", type=int, default=100)
    parser.add_argument("--ivf_probe", type=int, default=5)
    
    args = parser.parse_args()
    
    # Handle .dat vs .fvecs paths
    if args.database.endswith(".dat"):
        raw_db_path = args.database.replace(".dat", ".fvecs")
        id_mapping_file = args.database.replace(".dat", "_ids.txt")
    else:
        raw_db_path = args.database
        id_mapping_file = args.database.replace(".fvecs", "_ids.txt")
    
    if not os.path.exists(raw_db_path) or not os.path.exists(id_mapping_file):
        print(f"[ERR] Database or ID mapping missing. Run protein_embed.py first.")
        sys.exit(1)

    id_mapping = load_id_mapping(id_mapping_file)
    
    # Infer SwissProt FASTA if not provided
    if args.swissprot is None:
        possible_fasta = args.database.replace(".dat", ".fasta").replace(".fvecs", ".fasta").replace("_vectors", "")
        if os.path.exists(possible_fasta):
            args.swissprot = possible_fasta
            print(f"[INFO] Using inferred FASTA: {args.swissprot}")
        else:
            print("[ERR] Cannot find FASTA. Use -s to specify it.")
            sys.exit(1)
    
    # 1. Embed Query Sequences
    query_vecs, query_ids = embed_queries(args.queries)
    if len(query_vecs) == 0: sys.exit(1)
    
    # 2. Prepare Database (Normalization)
    normalized_db_path = "normalized_db_temp.fvecs"
    print("[INFO] Normalizing database vectors (Required for Cosine Similarity)...")
    
    raw_db = load_fvecs(raw_db_path)
    norms = np.linalg.norm(raw_db, axis=1, keepdims=True)
    norm_db = raw_db / np.maximum(norms, 1e-10)
    write_fvecs(normalized_db_path, norm_db)
    # Save batch queries
    batch_query_file = "all_queries_batch.fvecs"
    write_fvecs(batch_query_file, query_vecs)
    
    # Load sequences for identity check
    query_seqs = {r.id: str(r.seq) for r in SeqIO.parse(args.queries, "fasta")}
    db_seqs, db_descs = load_seq_db(args.swissprot)
    blast_db = setup_blast_db(args.swissprot)
    
    # Determine methods
    methods = ["lsh", "hypercube", "ivf", "ivfpq", "neural"] if args.method == "all" else [args.method]
    if "neural" in methods and not os.path.exists("nlsh_model.pth"): 
        print("[WARN] nlsh_model.pth missing. Skipping Neural.")
        methods.remove("neural")

    method_results = {}
    method_times = {}
    print("\n[START] Executing Benchmarks...")
    for m in methods:
        # Use NORMALIZED DB for everything
        res, t = run_c_search_batch(m, normalized_db_path, batch_query_file, args.N, vars(args), id_mapping, len(query_vecs))
        method_results[m] = res
        method_times[m] = t

    # 3. Generate Report
    f_out = open(args.output, 'w')
    
    for i, qid in enumerate(query_ids):
        if i % 10 == 0: print(f" -> Comparing Query {i+1}/{len(query_ids)}...")
        current_query_seq = query_seqs.get(qid, "")
        
        f_out.write("="*110 + "\n")
        f_out.write(f"\nQuery Protein: {qid}\n")
        f_out.write(f"N = {args.N} (Recall@N threshold)\n\n")
        
        # Ground Truth (BLAST)
        blast_hits, blast_t = run_blast(current_query_seq, qid, blast_db, args.N)
        blast_ids = set([h['id'] for h in blast_hits])
        
        f_out.write("[1] Method Comparison Summary\n")
        f_out.write("-" * 85 + "\n")
        f_out.write(f"{'Method':<17}| {'Time/query(s)':<15}| {'QPS':<7}| {'Recall@N vs BLAST':<20}\n")
        f_out.write("-" * 85 + "\n")
        
        qps_blast = int(1.0/blast_t) if blast_t > 0 else 0
        f_out.write(f"{'BLAST(Ref)':<17}| {blast_t:<15.3f}| {qps_blast:<7}| {1.0:<20.2f} (Reference)\n")
        
        for m in methods:
            res = method_results[m].get(i, [])
            t = method_times[m]
            qps = int(1.0/t) if t > 0.0001 else 0
            matches = 0
            for r in res:
                if check_match(blast_ids, r['id']): matches += 1
            recall = matches / len(blast_ids) if blast_ids else 0.0
            f_out.write(f"{m:<17}| {t:<15.4f}| {qps:<7}| {recall:<20.3f}\n")
        f_out.write("-" * 85 + "\n\n")

        f_out.write(f"[2] Top-N Neighbors per Method (Showing Top-10)\n")
        for m in methods:
            res = method_results[m].get(i, [])
            f_out.write(f"\nMethod: {m}\n")
            f_out.write(f"{'Rank':<6}| {'Neighbor ID':<20} | {'L2Dist':<11}| {'Ident.':<9}| {'In BLAST?':<11}| {'Bio Comment'}\n")
            f_out.write("-" * 125 + "\n")
            
            for idx, r in enumerate(res[:10]):
                in_blast = check_match(blast_ids, r['id'])
                is_match = "Yes" if in_blast else "No"
                d_val = r.get('dist', -1.0)
                
                found_id = r['id']
                found_seq = db_seqs.get(found_id, "")
                if found_seq:
                    ident_val = calc_identity(current_query_seq, found_seq)
                    ident_str = f"{ident_val:.1f}%"
                else:
                    ident_str = "N/A"
                
                # BIOLOGICAL RELEVANCE 
                clean_name = db_descs.get(found_id, "-")
                note = ""
                # Heuristic: Not in BLAST + Low Distance = Potential Remote Homolog
                if not in_blast and d_val < 0.65 and d_val > 0:
                    note = " [Possible Remote Homolog?]"
                
                bio_comment = f"{clean_name}{note}"
                if len(bio_comment) > 45: bio_comment = bio_comment[:42] + "..."
                disp_id = r['id'] if len(r['id']) <= 20 else r['id'][:17] + "..."
                
                f_out.write(f"{idx+1:<6}| {disp_id:<20} | {d_val:<11.4f}| {ident_str:<9}| {is_match:<11}| {bio_comment}\n")
        
        f_out.write("\n")
        f_out.flush()
        
    f_out.close()
    
    # Cleanup
    if os.path.exists(batch_query_file): os.remove(batch_query_file)
    if os.path.exists(normalized_db_path): os.remove(normalized_db_path)
    print(f"[DONE] Results saved to {args.output}")

if __name__ == "__main__":
    main()
