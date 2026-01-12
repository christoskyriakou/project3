# Protein Remote Homology Detection using ANN Methods

**Εργασία 3 - Υπολογιστική Βιολογία & Αναζήτηση Δεδομένων**

Σύστημα για την ανίχνευση απομακρυσμένων ομολόγων πρωτεϊνών χρησιμοποιώντας ESM-2 embeddings και Approximate Nearest Neighbor (ANN) αλγορίθμους.

---

## Περιγραφή

Το έργο αυτό αντιμετωπίζει το πρόβλημα της ανίχνευσης **remote homologs** - πρωτεϊνών με παρόμοια δομή και λειτουργία αλλά χαμηλή ομοιότητα ακολουθίας (<30%, "Twilight Zone"). 

### Βασικά Χαρακτηριστικά:
- Παραγωγή διανυσματικών αναπαραστάσεων με **ESM-2** (facebook/esm2_t6_8M_UR50D)
- Υποστήριξη **5 ANN αλγορίθμων**: LSH, Hypercube, IVF-Flat, IVFPQ, Neural LSH
- Σύγκριση με **BLAST** για βιολογική αξιολόγηση
- Υπολογισμός **Recall@N** και **QPS** μετρικών

---

## Δομή Αρχείων

```
.
├── protein_embed.py         # Παραγωγή ESM-2 embeddings
├── protein_search.py        # ANN benchmark & σύγκριση
├── dataset_parser.py        # Φόρτωση δεδομένων (από Εργασία 1/2)
├── distances.py             # Υπολογισμός αποστάσεων
├── models.py                # Neural LSH model
├── nlsh_build.py            # Neural LSH training
├── nlsh_search.py           # Neural LSH search
├── graph_utils.py           # KNN graph construction
├── lsh.c / lsh.h            # LSH C implementation
├── hypercube.c / hc.h       # Hypercube C implementation
├── ivfflat.c / kmeans.h     # IVF-Flat implementation
├── ivfpq.c / dataload.h     # IVFPQ implementation
├── requirements.txt         # Python dependencies
└── README.md                # Αυτό το αρχείο
```

---

## Εγκατάσταση

### Προαπαιτούμενα
- Python 3.10+
- CUDA (προαιρετικό, για GPU acceleration)
- GCC compiler (για C modules)
- BLAST+ tools

### Εγκατάσταση Python Dependencies

```bash
pip install -r requirements.txt
```

ή με conda:

```bash
conda create -n protein_search python=3.10
conda activate protein_search
pip install -r requirements.txt
```

### Εγκατάσταση BLAST

**Ubuntu/Debian:**
```bash
sudo apt-get install ncbi-blast+
```

**macOS:**
```bash
brew install blast
```

### Compilation C Modules

```bash
# LSH
gcc -o lsh_search lsh.c -lm -O3

# Hypercube
gcc -o hypercube_search hypercube.c -lm -O3

# IVF-Flat
gcc -o ivfflat_search ivfflat.c kmeans.c dataload.c -lm -O3

# IVFPQ
gcc -o ivfpq_search ivfpq.c kmeans.c dataload.c -lm -O3
```

---

## Χρήση

### 🚀 Quick Start (Πλήρες Pipeline)

```bash
# Make script executable
chmod +x run_protein_search.sh

# Run complete pipeline
./run_protein_search.sh \
    --data swissprot_small_small.fasta \
    --query targets.fasta \
    --output results.txt \
    --method all \
    -N 50
```

Αυτό θα εκτελέσει:
1. ESM-2 embedding generation
2. Neural LSH index building (αν χρειάζεται)
3. ANN search με όλες τις μεθόδους
4. Σύγκριση με BLAST

---

### Σενάριο 1: Παραγωγή Embeddings

```bash
python protein_embed.py \
    -i swissprot.fasta \
    -o protein_vectors.dat
```

**Παράμετροι:**
- `-i, --input`: Input FASTA αρχείο με πρωτεΐνες
- `-o, --output`: Output αρχείο (.fvecs ή .dat)
- `--model`: ESM-2 model (default: facebook/esm2_t6_8M_UR50D)
- `--batch_size`: Batch size για GPU (default: 8)

**Έξοδος:**
- `protein_vectors.fvecs`: Embeddings σε fvecs format
- `protein_vectors_ids.txt`: Mapping index → sequence ID

---

### Σενάριο 2: Build Neural LSH Index

```bash
python protein_nlsh_build.py \
    -d protein_vectors.fvecs \
    -i ./protein_index \
    --knn 10 \
    -m 100 \
    --epochs 20
```

**Παράμετροι:**
- `-d, --data`: Protein embeddings (.fvecs)
- `-i, --index`: Output index directory
- `--knn`: k for KNN graph (default: 10)
- `-m`: Number of partitions (default: 100)
- `--epochs`: Training epochs (default: 20)

---

### Σενάριο 3: ANN Search Benchmark

```bash
python protein_search.py \
    -d protein_vectors.dat \
    -q targets.fasta \
    -o results.txt \
    -method all \
    -N 50
```

**Παράμετροι:**
- `-d, --data`: Embedding data file (.fvecs)
- `-q, --query`: Query FASTA file
- `-o, --output`: Output results file
- `-method`: ANN method (`all`, `lsh`, `hypercube`, `neural`, `ivf`, `ivfpq`)
- `-N`: Number of neighbors (default: 50)

**Έξοδος:**

Για κάθε query, το αρχείο περιέχει:

1. **Συνοπτική σύγκριση**: QPS και Recall@N για κάθε μέθοδο
2. **Top-N γείτονες**: Αναλυτική λίστα με:
   - Neighbor ID
   - L2 distance
   - BLAST identity (%)
   - In BLAST Top-N? (Yes/No)
   - Bio comment (π.χ. "Remote homolog?")

### Σενάριο 4: Analyze Results

```bash
# Basic analysis
python analyze_results.py -i results.txt

# With plots
python analyze_results.py -i results.txt --plot --output-dir ./plots
```

Αυτό θα παράγει:
- Summary statistics table
- Recall vs QPS plots
- Method comparison charts
- List of potential remote homologs

---

## Προετοιμασία για C Executables

Πριν τρέξετε το search, βεβαιωθείτε ότι έχετε μεταγλωττίσει το `./search` executable:

```bash
# Στο root directory του project
make

# Ή χειροκίνητα
cd LSH_Project && make && cd ..
cd HYPERCUBE_Project && make && cd ..
cd IVFFlat && make && cd ..
cd IVFPQ && make && cd ..
```

Το `protein_search.py` περιμένει να βρει το `./search` executable στο working directory.

```
Query Protein: sp|Q6GZX4|001R_FRG3G
N = 50 (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)

[1] Συνοπτική σύγκριση μεθόδων
------------------------------------------------------------------------------
Method               | Time/query (s)  | QPS        | Recall@N vs BLAST Top-N
------------------------------------------------------------------------------
lsh                  | 0.025           | 40         | 0.88
hypercube            | 0.032           | 31         | 0.84
neural               | 0.012           | 83         | 0.92
ivf                  | 0.010           | 100        | 0.90
ivfpq                | 0.007           | 143        | 0.86
BLAST (Ref)          | 1.450           | 0.7        | 1.00 (ορίζει το Top-N)
------------------------------------------------------------------------------

[2] Top-N γείτονες ανά μέθοδο (εδώ π.χ. N = 10 για εκτύπωση)

Method: neural
Rank   | Neighbor ID          | L2 Dist    | BLAST Identity  | In BLAST Top-N?   | Bio comment
--------------------------------------------------------------------------------------------------------------
1      | sp|Q6GZX3|002L       | 0.145      | 18.5            | Yes               | Remote homolog? (Twilight Zone)
2      | sp|Q197F8|002R       | 0.167      | 24.2            | Yes               | Remote homolog? (Twilight Zone)
...
```

---

## Αλγόριθμοι & Υπερπαράμετροι

### 1. Euclidean LSH
- **k**: Number of hash functions per table (προτεινόμενο: 10-14)
- **L**: Number of hash tables (προτεινόμενο: 8-12)
- **w**: Bucket width (προτεινόμενο: 4.0 για ESM-2 embeddings)

### 2. Hypercube Projection
- **k**: Projection dimensions (προτεινόμενο: 12-16)
- **M**: Max candidates (προτεινόμενο: 5000-10000)
- **probes**: Number of vertices to probe (προτεινόμενο: 50-100)

### 3. IVF-Flat
- **kclusters**: Number of clusters (προτεινόμενο: √n)
- **nprobe**: Clusters to search (προτεινόμενο: 10-20)

### 4. IVFPQ
- **kclusters**: Coarse quantizer clusters (προτεινόμενο: √n)
- **M**: Number of subspaces (προτεινόμενο: 8-16)
- **nbits**: Bits per subspace (προτεινόμενο: 8)
- **nprobe**: Clusters to search (προτεινόμενο: 15-25)

### 5. Neural LSH
- **m**: Number of partitions (προτεινόμενο: 100-200)
- **T**: Multi-probe parameter (προτεινόμενο: 5-10)
- **layers**: MLP depth (προτεινόμενο: 3)
- **hidden**: Hidden layer size (προτεινόμενο: 64-128)

---

## Βιολογική Αξιολόγηση

### Ορισμός Remote Homolog

Θεωρούμε μία πρωτεΐνη ως **υποψήφια remote homolog** όταν:

1. **BLAST identity < 30%** (Twilight Zone)
2. **Μικρή L2 απόσταση** στο embedding space (Top-N)
3. **Κοινά χαρακτηριστικά**:
   - Ίδια Pfam domain
   - Παρόμοιοι GO terms
   - Ίδιος EC number
   - Κοινή λειτουργική οικογένεια

### Χρήση UniProt Annotations

Για την επαλήθευση remote homologs:

1. Ανάκτηση UniProt entries για τους γείτονες
2. Έλεγχος για:
   - Function annotations
   - Pfam domains (InterPro)
   - GO terms (Molecular Function, Biological Process)
   - EC numbers (enzymatic activity)

```python
# Παράδειγμα annotation check
from Bio import Entrez, SwissProt

def check_homology(seq_id1, seq_id2):
    # Retrieve UniProt records
    record1 = get_uniprot_record(seq_id1)
    record2 = get_uniprot_record(seq_id2)
    
    # Check for common domains
    domains1 = get_pfam_domains(record1)
    domains2 = get_pfam_domains(record2)
    
    common_domains = set(domains1) & set(domains2)
    
    return len(common_domains) > 0
```

---

## Αποτελέσματα & Ανάλυση

### Αναμενόμενα Αποτελέσματα

| Method     | QPS  | Recall@50 | Trade-off            |
|------------|------|-----------|----------------------|
| Neural LSH | 80+  | 0.90-0.95 | Καλύτερη ισορροπία   |
| IVF-Flat   | 100+ | 0.88-0.92 | Πιο γρήγορο          |
| IVFPQ      | 150+ | 0.82-0.88 | Ταχύτερο, λιγότερο ακριβές |
| LSH        | 40+  | 0.85-0.90 | Καλό για high-dim    |
| Hypercube  | 30+  | 0.80-0.85 | Πιο αργό             |

### Remote Homolog Detection

Τα embedding-based methods ξεπερνούν το BLAST σε:
- **Twilight Zone** (15-30% identity)
- Δομικές ομολογίες χωρίς sequence conservation
- Cross-family functional relationships

---

## Troubleshooting

### C Executable Not Found

```bash
Error: ./search: No such file or directory
```

**Λύση:**
```bash
# Compile the main search program
make

# Or create symbolic link to your executable
ln -s LSH_Project/lsh ./search
```

### Out of Memory (GPU)

Μειώστε το `--batch_size`:
```bash
python protein_embed.py -i input.fasta -o output.dat --batch_size 4
```

### Αργή BLAST

Περιορίστε το `-max_target_seqs`:
```bash
blastp -query q.fasta -db db -max_target_seqs 100
```

### C Module Compilation Errors

Βεβαιωθείτε ότι έχετε εγκαταστήσει:
```bash
sudo apt-get install build-essential
```

---

## Citation

Εάν χρησιμοποιήσετε αυτό το έργο:

```bibtex
@software{protein_ann_search_2025,
  title={Protein Remote Homology Detection using ANN Methods},
  author={[Το Όνομά σας]},
  year={2025},
  institution={[Πανεπιστήμιο]}
}
```

### ESM-2 Model:
```bibtex
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and others},
  journal={bioRxiv},
  year={2022}
}
```

---

## License

MIT License - Ελεύθερο για εκπαιδευτική και ερευνητική χρήση.

---

## Επικοινωνία

Για ερωτήσεις ή βοήθεια:
- Email: [your-email]
- GitHub Issues: [repository-url]

---

**Καλή επιτυχία στην εργασία σας! 🧬🔬**