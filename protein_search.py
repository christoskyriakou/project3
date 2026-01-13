#!/usr/bin/env python3
"""
protein_search.py
Comprehensive ANN benchmark for protein remote homology detection
"""

import argparse
import numpy as np
import time
import subprocess
import os
import tempfile
import struct
from Bio import SeqIO
from collections import defaultdict

from dataset_parser import load_dataset
from distances import L2_distance_batch

# Import your existing ANN methods
import ctypes
import sys


class ProteinSearchBenchmark:
    def __init__(self, data_path, query_fasta, output_path, method="all", N=50, 
                 index_dir=None, params=None):
        self.data_path = data_path
        self.query_fasta = query_fasta
        self.output_path = output_path
        self.method = method
        self.N = N
        self.index_dir = index_dir or "./protein_index"
        self.params = params or {}
        
        # Load data
        print("[SEARCH] Loading protein embeddings...")
        self.X, dtype = load_dataset(data_path)
        self.n, self.dim = self.X.shape
        print(f"[SEARCH] Loaded {self.n} vectors, dim={self.dim}")
        
        # Convert embeddings to temporary files for C programs
        self.setup_temp_files()
        
        # Load ID mapping
        self.load_id_mapping()
        
        # Load queries
        print("[SEARCH] Loading query sequences...")
        self.load_queries()
        
        # BLAST setup
        self.setup_blast()
    
    def setup_temp_files(self):
        """Create temporary .fvecs files for C programs"""

        
        # Data file
        self.temp_data = tempfile.NamedTemporaryFile(suffix='.fvecs', delete=False)
        with open(self.temp_data.name, 'wb') as f:
            for i in range(self.n):
                f.write(struct.pack('i', self.dim))
                f.write(self.X[i].astype(np.float32).tobytes())
        
        print(f"[SEARCH] Created temp data file: {self.temp_data.name}")
    
    def save_query_fvecs(self, query):
        """Save single query to temporary .fvecs file"""
        query_file = tempfile.NamedTemporaryFile(suffix='.fvecs', delete=False)
        
        with open(query_file.name, 'wb') as f:
            f.write(struct.pack('i', self.dim))
            f.write(query.astype(np.float32).tobytes())
        
        return query_file.name
    
    def cleanup_temp_files(self):
        """Cleanup temporary files"""
        try:
            os.unlink(self.temp_data.name)
        except:
            pass

    def load_id_mapping(self):
        """Load sequence ID to index mapping"""
        # Preferred mapping file: <data_basename>_ids.txt
        mapping_path = self.data_path.replace(".fvecs", "_ids.txt").replace(".dat", "_ids.txt")

        self.id_to_idx = {}
        self.idx_to_id = {}

        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                    idx_str, seq_id = parts
                    idx = int(idx_str)
                    self.id_to_idx[seq_id] = idx
                    self.idx_to_id[idx] = seq_id
            print(f"[SEARCH] Loaded {len(self.id_to_idx)} ID mappings from {mapping_path}")
        else:
            # Fallback: if you have ids.txt (one ID per line), use that
            if os.path.exists("ids.txt"):
                with open("ids.txt", "r") as f:
                    ids = [line.strip() for line in f if line.strip()]
                for i, seq_id in enumerate(ids[:self.n]):
                    self.id_to_idx[seq_id] = i
                    self.idx_to_id[i] = seq_id
                print(f"[SEARCH] Loaded {len(self.id_to_idx)} ID mappings from ids.txt")
            else:
                print("[SEARCH] Warning: No ID mapping found, using indices")
                for i in range(self.n):
                    self.idx_to_id[i] = f"seq_{i}"

    
    def load_queries(self):
        """Load query sequences and generate embeddings"""
        from protein_embed import generate_embeddings
        
        # Generate query embeddings
        self.Q, self.query_ids = generate_embeddings(self.query_fasta, batch_size=4)
        print(f"[SEARCH] Generated {len(self.query_ids)} query embeddings")
    
    def setup_blast(self):
        """Prepare BLAST database"""
        print("[SEARCH] Setting up BLAST...")
        
        # Find original FASTA (assume it's in same dir as embeddings)
        base_path = self.data_path.replace(".fvecs", "").replace(".dat", "")
        possible_fasta = [
            base_path + ".fasta",
            base_path + ".fa",
            "swissprot.fasta",
            "swissprot_small.fasta"
        ]
        
        self.blast_db_fasta = None
        for path in possible_fasta:
            if os.path.exists(path):
                self.blast_db_fasta = path
                break
        
        if self.blast_db_fasta is None:
            print("[SEARCH] Warning: BLAST database FASTA not found, BLAST comparison disabled")
            self.blast_enabled = False
            return
        
        # Create BLAST database
        self.blast_db = self.blast_db_fasta.replace(".fasta", "_blastdb")
        
        if not os.path.exists(self.blast_db + ".phr"):
            print(f"[SEARCH] Creating BLAST database from {self.blast_db_fasta}")
            subprocess.run([
                "makeblastdb",
                "-in", self.blast_db_fasta,
                "-dbtype", "prot",
                "-out", self.blast_db
            ], check=True)
        
        self.blast_enabled = True
        print("[SEARCH] BLAST setup complete")
    
    def run_blast(self, query_seq, query_id):
        """Run BLAST for a single query"""
        if not self.blast_enabled:
            return []
        
        # Create temp query file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">{query_id}\n{query_seq}\n")
            temp_query = f.name
        
        # Run BLAST
        blast_out = temp_query + ".blast"
        
        try:
            subprocess.run([
                "blastp",
                "-query", temp_query,
                "-db", self.blast_db,
                "-out", blast_out,
                "-outfmt", "6 sseqid pident evalue bitscore",
                "-max_target_seqs", str(self.N * 2),
                "-evalue", "10"
            ], check=True, capture_output=True)
            
            # Parse results
            blast_results = []
            with open(blast_out, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        sseqid = parts[0].split("|")[1] if "|" in parts[0] else parts[0]
                        pident = float(parts[1])
                        evalue = float(parts[2])
                        bitscore = float(parts[3])
                        blast_results.append({
                            'id': sseqid,
                            'identity': pident,
                            'evalue': evalue,
                            'bitscore': bitscore
                        })
            
            # Cleanup
            os.remove(temp_query)
            os.remove(blast_out)
            
            return blast_results
        
        except Exception as e:
            print(f"[SEARCH] BLAST error: {e}")
            return []
    
    def search_lsh(self, query):
        """LSH search using C implementation"""
        query_file = self.save_query_fvecs(query)
        output_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        
        # Default LSH parameters (tuned for protein embeddings)
        k = self.params.get('lsh_k', 12)
        L = self.params.get('lsh_L', 10)
        w = self.params.get('lsh_w', 4.0)
        
        cmd = [
            "./search",
            "-d", self.temp_data.name,
            "-q", query_file,
            "-type", "sift",  # Use sift format (float32)
            "-k", str(k),
            "-L", str(L),
            "-w", str(w),
            "-N", str(self.N),
            "-R", "10000",  # Large radius for proteins
            "-o", output_file.name,
            "-range", "false",
            "-lsh"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[SEARCH] LSH warning: {result.stderr}")
                return self.brute_force_search(query)
            
            # Parse output
            results = self.parse_c_output(output_file.name)
            
            # Cleanup
            os.unlink(query_file)
            os.unlink(output_file.name)
            
            return results
        
        except Exception as e:
            print(f"[SEARCH] LSH error: {e}, falling back to brute force")
            return self.brute_force_search(query)
    
    def search_hypercube(self, query):
        """Hypercube search using C implementation"""
        query_file = self.save_query_fvecs(query)
        output_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        
        # Default Hypercube parameters
        kproj = self.params.get('hc_kproj', 14)
        w = self.params.get('hc_w', 4.0)
        M = self.params.get('hc_M', 10000)
        probes = self.params.get('hc_probes', 100)
        
        cmd = [
            "./search",
            "-d", self.temp_data.name,
            "-q", query_file,
            "-type", "sift",
            "-kproj", str(kproj),
            "-w", str(w),
            "-M", str(M),
            "-probes", str(probes),
            "-N", str(self.N),
            "-R", "10000",
            "-o", output_file.name,
            "-range", "false",
            "-hypercube"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[SEARCH] Hypercube warning: {result.stderr}")
                return self.brute_force_search(query)
            
            results = self.parse_c_output(output_file.name)
            
            os.unlink(query_file)
            os.unlink(output_file.name)
            
            return results
        
        except Exception as e:
            print(f"[SEARCH] Hypercube error: {e}, falling back to brute force")
            return self.brute_force_search(query)
    
    def search_neural(self, query):
        """Neural LSH search"""
        try:
            # Check if Neural LSH index exists
            if not os.path.exists(self.index_dir):
                print(f"[SEARCH] Neural LSH index not found at {self.index_dir}")
                print(f"[SEARCH] Please run nlsh_build.py first")
                return self.brute_force_search(query)
            
            # Import Neural LSH modules
            from models import MLP
            from nlsh_search import load_index, multi_probe_search, knn_search
            
            # Load index
            T = self.params.get('neural_T', 5)
            m = self.params.get('neural_m', 100)
            layers = self.params.get('neural_layers', 3)
            nodes = self.params.get('neural_nodes', 64)
            
            model, inverted, device = load_index(
                self.index_dir, 
                self.dim, 
                m, 
                layers, 
                nodes
            )
            
            # Multi-probe search
            candidate_indices = multi_probe_search(model, inverted, query, T, device)
            
            # KNN search on candidates
            if len(candidate_indices) == 0:
                return []
            
            candidates_data = self.X[candidate_indices]
            distances = L2_distance_batch(candidates_data, query)
            
            sorted_idx = np.argsort(distances)[:self.N]
            
            results = []
            for i in sorted_idx:
                idx = candidate_indices[i]
                results.append({
                    'index': int(idx),
                    'id': self.idx_to_id.get(idx, f"seq_{idx}"),
                    'distance': float(distances[i])
                })
            
            return results
        
        except Exception as e:
            print(f"[SEARCH] Neural LSH error: {e}, falling back to brute force")
            import traceback
            traceback.print_exc()
            return self.brute_force_search(query)
    
    def search_ivf(self, query):
        """IVF-Flat search using C implementation"""
        query_file = self.save_query_fvecs(query)
        output_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        
        # Default IVF parameters
        kclusters = self.params.get('ivf_kclusters', int(np.sqrt(self.n)))
        nprobe = self.params.get('ivf_nprobe', 15)
        
        cmd = [
            "./search",
            "-d", self.temp_data.name,
            "-q", query_file,
            "-type", "sift",
            "-kclusters", str(kclusters),
            "-nprobe", str(nprobe),
            "-N", str(self.N),
            "-R", "10000",
            "-o", output_file.name,
            "-range", "false",
            "-ivfflat"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"[SEARCH] IVF warning: {result.stderr}")
                return self.brute_force_search(query)
            
            results = self.parse_c_output(output_file.name)
            
            os.unlink(query_file)
            os.unlink(output_file.name)
            
            return results
        
        except Exception as e:
            print(f"[SEARCH] IVF error: {e}, falling back to brute force")
            return self.brute_force_search(query)
    
    def search_ivfpq(self, query):
        """IVFPQ search using C implementation"""
        query_file = self.save_query_fvecs(query)
        output_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        
        # Default IVFPQ parameters
        kclusters = self.params.get('ivfpq_kclusters', int(np.sqrt(self.n)))
        nprobe = self.params.get('ivfpq_nprobe', 20)
        M = self.params.get('ivfpq_M', 16)
        nbits = self.params.get('ivfpq_nbits', 8)
        
        cmd = [
            "./search",
            "-d", self.temp_data.name,
            "-q", query_file,
            "-type", "sift",
            "-kclusters", str(kclusters),
            "-nprobe", str(nprobe),
            "-M", str(M),
            "-nbits", str(nbits),
            "-N", str(self.N),
            "-R", "10000",
            "-o", output_file.name,
            "-range", "false",
            "-ivfpq"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if result.returncode != 0:
                print(f"[SEARCH] IVFPQ warning: {result.stderr}")
                return self.brute_force_search(query)
            
            results = self.parse_c_output(output_file.name)
            
            os.unlink(query_file)
            os.unlink(output_file.name)
            
            return results
        
        except Exception as e:
            print(f"[SEARCH] IVFPQ error: {e}, falling back to brute force")
            return self.brute_force_search(query)
    
    def parse_c_output(self, output_file):
        """Parse C program output file"""
        results = []

        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if line.startswith("Nearest neighbor-"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        idx_str = parts[1].strip()

                        if "image_id_" in idx_str:
                            idx = int(idx_str.replace("image_id_", ""))
                        else:
                            idx = int(idx_str)

                        i += 1
                        if i < len(lines) and "distanceApproximate:" in lines[i]:
                            dist_line = lines[i].strip()
                            dist = float(dist_line.split(":")[1].strip())

                            results.append({
                                'index': idx,
                                'id': self.idx_to_id.get(idx, f"seq_{idx}"),
                                'distance': dist
                            })

                i += 1

            return results

        except Exception as e:
            print(f"[SEARCH] Error parsing C output: {e}")
            return []


    def brute_force_search(self, query):
        """Exact nearest neighbor search (fallback)"""
        distances = L2_distance_batch(self.X, query)
        sorted_idx = np.argsort(distances)[:self.N]

        results = []
        for idx in sorted_idx:
            idx = int(idx)
            results.append({
                'index': idx,
                'id': self.idx_to_id.get(idx, f"seq_{idx}"),
                'distance': float(distances[idx])
            })

        return results

    def compute_recall(self, ann_results, blast_results):
        """Compute Recall@N vs BLAST"""
        if not blast_results:
            return 0.0
        
        blast_ids = set([r['id'] for r in blast_results[:self.N]])
        ann_ids = set([r['id'] for r in ann_results[:self.N]])
        
        intersection = len(blast_ids & ann_ids)
        recall = intersection / len(blast_ids) if len(blast_ids) > 0 else 0.0
        
        return recall
    
    def annotate_result(self, seq_id, blast_identity):
        """Add biological annotation to result"""
        # Check for remote homolog criteria
        if blast_identity < 30 and blast_identity > 0:
            return "Remote homolog? (Twilight Zone)"
        elif blast_identity == 0:
            return "Not in BLAST results"
        else:
            return "--"
    
    def run_benchmark(self):
        """Run full benchmark"""
        results = []
        
        methods = {
            'lsh': self.search_lsh,
            'hypercube': self.search_hypercube,
            'neural': self.search_neural,
            'ivf': self.search_ivf,
            'ivfpq': self.search_ivfpq
        }
        
        if self.method != "all":
            methods = {self.method: methods[self.method]}
        
        # Get query sequences for BLAST
        query_seqs = {}
        for record in SeqIO.parse(self.query_fasta, "fasta"):
            query_seqs[record.id] = str(record.seq)
        
        # Process each query
        for q_idx, (query, query_id) in enumerate(zip(self.Q, self.query_ids)):
            print(f"\n[SEARCH] Processing query {q_idx + 1}/{len(self.Q)}: {query_id}")
            
            query_result = {
                'id': query_id,
                'index': q_idx,
                'methods': {}
            }
            
            # Run BLAST
            blast_results = []
            blast_time = 0
            if self.blast_enabled and query_id in query_seqs:
                t_start = time.time()
                blast_results = self.run_blast(query_seqs[query_id], query_id)
                blast_time = time.time() - t_start
                print(f"[SEARCH] BLAST: {len(blast_results)} results in {blast_time:.3f}s")
            
            query_result['blast'] = {
                'results': blast_results,
                'time': blast_time
            }
            
            # Run each ANN method
            for method_name, method_func in methods.items():
                print(f"[SEARCH] Running {method_name}...")
                
                t_start = time.time()
                ann_results = method_func(query)
                method_time = time.time() - t_start
                
                # Ensure we have exactly N results
                if len(ann_results) < self.N:
                    print(f"[SEARCH] Warning: {method_name} returned only {len(ann_results)} results")
                
                # Compute metrics
                recall = self.compute_recall(ann_results, blast_results)
                qps = 1.0 / method_time if method_time > 0 else 0
                
                # Annotate results
                for res in ann_results[:min(10, len(ann_results))]:  # Annotate top-10
                    blast_match = next((b for b in blast_results if b['id'] == res['id']), None)
                    res['blast_identity'] = blast_match['identity'] if blast_match else 0
                    res['in_blast_topn'] = blast_match is not None and blast_results.index(blast_match) < self.N
                    res['bio_comment'] = self.annotate_result(res['id'], res['blast_identity'])
                
                query_result['methods'][method_name] = {
                    'results': ann_results,
                    'time': method_time,
                    'qps': qps,
                    'recall': recall
                }
                
                print(f"[SEARCH] {method_name}: Recall@{self.N}={recall:.3f}, QPS={qps:.1f}")
            
            results.append(query_result)
        
        # Cleanup
        self.cleanup_temp_files()
        
        return results
    
    def write_results(self, results):
        """Write formatted results to output file"""
        with open(self.output_path, 'w') as f:
            for query_result in results:
                f.write(f"\nQuery Protein: {query_result['id']}\n")
                f.write(f"N = {self.N} (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)\n\n")
                
                # [1] Summary comparison
                f.write("[1] Συνοπτική σύγκριση μεθόδων\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Method':<20} | {'Time/query (s)':<15} | {'QPS':<10} | {'Recall@N vs BLAST Top-N':<20}\n")
                f.write("-" * 90 + "\n")
                
                for method_name, method_data in query_result['methods'].items():
                    f.write(f"{method_name:<20} | {method_data['time']:<15.3f} | {method_data['qps']:<10.1f} | {method_data['recall']:<20.3f}\n")
                
                if query_result['blast']['results']:
                    blast_time = query_result['blast']['time']
                    blast_qps = 1.0 / blast_time if blast_time > 0 else 0
                    f.write(f"{'BLAST (Ref)':<20} | {blast_time:<15.3f} | {blast_qps:<10.1f} | {'1.00 (ορίζει το Top-N)':<20}\n")
                
                f.write("-" * 90 + "\n\n")
                
                # [2] Detailed Top-N per method
                f.write(f"[2] Top-N γείτονες ανά μέθοδο (εδώ π.χ. N = 10 για εκτύπωση)\n\n")
                
                for method_name, method_data in query_result['methods'].items():
                    f.write(f"Method: {method_name}\n")
                    f.write(f"{'Rank':<6} | {'Neighbor ID':<20} | {'L2 Dist':<10} | {'BLAST Identity':<15} | {'In BLAST Top-N?':<17} | {'Bio comment':<30}\n")
                    f.write("-" * 110 + "\n")
                    
                    for rank, res in enumerate(method_data['results'][:10], 1):
                        f.write(f"{rank:<6} | {res['id']:<20} | {res['distance']:<10.3f} | "
                                f"{res['blast_identity']:<15.1f} | "
                                f"{'Yes' if res['in_blast_topn'] else 'No':<17} | "
                                f"{res['bio_comment']:<30}\n")
                    
                    f.write("\n")
                
                f.write("\n" + "=" * 110 + "\n")
        
        print(f"[SEARCH] Results written to {self.output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Protein Remote Homology Detection - ANN Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all methods
  python protein_search.py -d swissprot.fvecs -q queries.fasta -o results.txt
  
  # Test only LSH
  python protein_search.py -d swissprot.fvecs -q queries.fasta -o results.txt -method lsh
  
  # Custom parameters
  python protein_search.py -d swissprot.fvecs -q queries.fasta -o results.txt \\
    --lsh_k 15 --lsh_L 12 --ivf_nprobe 20
        """
    )
    
    # Required arguments
    parser.add_argument("-d", "--data", required=True, 
                       help="Embedding data file (.fvecs)")
    parser.add_argument("-q", "--query", required=True, 
                       help="Query FASTA file")
    parser.add_argument("-o", "--output", required=True, 
                       help="Output results file")
    
    # Method selection
    parser.add_argument("-method", 
                       choices=["all", "lsh", "hypercube", "neural", "ivf", "ivfpq"],
                       default="all", 
                       help="ANN method to use (default: all)")
    parser.add_argument("-N", type=int, default=50, 
                       help="Number of nearest neighbors (default: 50)")
    parser.add_argument("--index", default="./protein_index", 
                       help="Neural LSH index directory")
    
    # LSH parameters
    lsh_group = parser.add_argument_group('LSH parameters')
    lsh_group.add_argument("--lsh_k", type=int, default=12, 
                          help="Hash functions per table (default: 12)")
    lsh_group.add_argument("--lsh_L", type=int, default=10, 
                          help="Number of hash tables (default: 10)")
    lsh_group.add_argument("--lsh_w", type=float, default=4.0, 
                          help="Bucket width (default: 4.0)")
    
    # Hypercube parameters
    hc_group = parser.add_argument_group('Hypercube parameters')
    hc_group.add_argument("--hc_kproj", type=int, default=14, 
                         help="Projection dimensions (default: 14)")
    hc_group.add_argument("--hc_w", type=float, default=4.0, 
                         help="Bucket width (default: 4.0)")
    hc_group.add_argument("--hc_M", type=int, default=10000, 
                         help="Max candidates (default: 10000)")
    hc_group.add_argument("--hc_probes", type=int, default=100, 
                         help="Vertices to probe (default: 100)")
    
    # IVF parameters
    ivf_group = parser.add_argument_group('IVF parameters')
    ivf_group.add_argument("--ivf_kclusters", type=int, default=None, 
                          help="Number of clusters (default: sqrt(n))")
    ivf_group.add_argument("--ivf_nprobe", type=int, default=15, 
                          help="Clusters to search (default: 15)")
    
    # IVFPQ parameters
    ivfpq_group = parser.add_argument_group('IVFPQ parameters')
    ivfpq_group.add_argument("--ivfpq_kclusters", type=int, default=None, 
                            help="Number of clusters (default: sqrt(n))")
    ivfpq_group.add_argument("--ivfpq_nprobe", type=int, default=20, 
                            help="Clusters to search (default: 20)")
    ivfpq_group.add_argument("--ivfpq_M", type=int, default=16, 
                            help="Number of subspaces (default: 16)")
    ivfpq_group.add_argument("--ivfpq_nbits", type=int, default=8, 
                            help="Bits per subspace (default: 8)")
    
    # Neural LSH parameters
    neural_group = parser.add_argument_group('Neural LSH parameters')
    neural_group.add_argument("--neural_m", type=int, default=100, 
                             help="Number of partitions (default: 100)")
    neural_group.add_argument("--neural_T", type=int, default=5, 
                             help="Multi-probe parameter (default: 5)")
    neural_group.add_argument("--neural_layers", type=int, default=3, 
                             help="MLP layers (default: 3)")
    neural_group.add_argument("--neural_nodes", type=int, default=64, 
                             help="Hidden layer size (default: 64)")
    
    args = parser.parse_args()
    
    # Collect parameters
    params = {
        'lsh_k': args.lsh_k,
        'lsh_L': args.lsh_L,
        'lsh_w': args.lsh_w,
        'hc_kproj': args.hc_kproj,
        'hc_w': args.hc_w,
        'hc_M': args.hc_M,
        'hc_probes': args.hc_probes,
        'ivf_kclusters': args.ivf_kclusters,
        'ivf_nprobe': args.ivf_nprobe,
        'ivfpq_kclusters': args.ivfpq_kclusters,
        'ivfpq_nprobe': args.ivfpq_nprobe,
        'ivfpq_M': args.ivfpq_M,
        'ivfpq_nbits': args.ivfpq_nbits,
        'neural_m': args.neural_m,
        'neural_T': args.neural_T,
        'neural_layers': args.neural_layers,
        'neural_nodes': args.neural_nodes
    }
    
    # Run benchmark
    benchmark = ProteinSearchBenchmark(
        data_path=args.data,
        query_fasta=args.query,
        output_path=args.output,
        method=args.method,
        N=args.N,
        index_dir=args.index,
        params=params
    )
    
    results = benchmark.run_benchmark()
    benchmark.write_results(results)
    
    print("[SEARCH] Benchmark complete!")


if __name__ == "__main__":
    main()