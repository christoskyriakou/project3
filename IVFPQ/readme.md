# IVFPQ: Inverted File Product Quantization Search

## Περιγραφή
Υλοποίηση του αλγορίθμου IVFPQ (Inverted File with Product Quantization) για 
προσεγγιστική αναζήτηση πλησιέστερων γειτόνων σε υψηλοδιάστατους χώρους.

## Χαρακτηριστικά
- Coarse quantization με k-means clustering
- Product Quantization για συμπίεση διανυσμάτων
- Asymmetric distance computation για γρήγορη αναζήτηση
- Υποστήριξη MNIST και SIFT datasets
- N-nearest neighbors search
- Range search (R-near neighbors)
- Μετρικές απόδοσης: AF, Recall@N, QPS, Silouhette 

## Δομή Αρχείων
ivfpq/
├── ivffpq.c/h        # ivfpq function implementation (create, build, search, free)
├── kmeans.c/h        # kmeans and utility function implementation (euclidean distance, vector ops, kmeans ops)
├── dataloading.c/h   # dataloading and utility functions implementation
├── search.c          # Main program, parsing arguments
├── Makefile          # Build automation
├── res/              # directory with minst/sift resources
└── README.md         # Τεκμηρίωση (αυτό το αρχείο)


## Μεταγλώττιση

```bash
make
```

Αυτό δημιουργεί το εκτελέσιμο `search`.

## Χρήση

### Βασική εντολή
```bash
./search -d <input_file> -q <query_file> -o <output_file> -type <mnist|sift> -ivfpq
```

### Παράμετροι
- `-d <file>` - Αρχείο dataset (υποχρεωτικό)
- `-q <file>` - Αρχείο queries (υποχρεωτικό)
- `-o <file>` - Αρχείο εξόδου (default: output.txt)
- `-type <mnist|sift>` - Τύπος dataset (default: mnist)
- `-kclusters <int>` - Αριθμός clusters για coarse quantizer (default: 50)
- `-nprobe <int>` - Πλήθος clusters προς έλεγχο (default: 5)
- `-M <int>` - Αριθμός υποδιανυσμάτων για PQ (default: 16)
- `-nbits <int>` - Bits για PQ codebook (default: 8, δίνει 256 clusters)
- `-N <int>` - Αριθμός πλησιέστερων γειτόνων (default: 1)
- `-R <float>` - Ακτίνα για range search (default: 2000 για MNIST, 2 για SIFT)
- `-seed <int>` - Random seed (default: 1)
- `-range <true|false>` - Ενεργοποίηση range search (default: true)
- `-ivfpq` - Flag για επιλογή μεθόδου

### Παραδείγματα

#### MNIST
```bash
./search -d res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte -o results.txt -type mnist -kclusters 50 -nprobe 10 -M 32 -nbits 8 -N 5 -R 2000 -ivfpq
```

#### SIFT
```bash
./search -d res/sift_base.fvecs -q res/sift_query.fvecs -o results.txt -type sift -kclusters 50 -nprobe 10 -M 8 -nbits 8 -N 10 -R 2.0 -ivfpq 
```

## Μορφή Εξόδου

```
IVFPQ
Query: 0
Nearest neighbor-1: 1234
distanceApproximate: 1234.567890
distanceTrue: 1230.123456
...
Nearest neighbor-N: 5678
distanceApproximate: 2345.678901
distanceTrue: 2340.234567
R-near neighbors:
100
234
567
...
Average AF: 1.003456
Recall@N: 0.987654
QPS: 123.45
tApproximateAverage: 0.008100
tTrueAverage: 1.234567
```

## Αλγόριθμος

### 1. Coarse Quantization
- K-means clustering του dataset σε `kclusters` clusters
- Δημιουργία inverted index: κάθε cluster κρατά λίστα διανυσμάτων

### 2. Product Quantization
- Διαίρεση κάθε διανύσματος σε M υποδιανύσματα
- K-means σε κάθε υποχώρο (2^nbits clusters)
- Κωδικοποίηση: κάθε διάνυσμα → M bytes

### 3. Αναζήτηση
- Βρες `nprobe` πλησιέστερα clusters στο query
- Για κάθε cluster:
  - Υπολόγισε asymmetric distance για κάθε κωδικοποιημένο διάνυσμα
  - Distance = Σ(dist(query_subvec, pq_centroid[code]))
- Επέστρεψε top-N αποτελέσματα

## Μετρικές Απόδοσης

- **AF (Approximation Factor)**: Μέσος λόγος απόστασης προσεγγιστικής / ακριβούς
- **Recall@N**: Ποσοστό σωστών αποτελεσμάτων στα top-N
- **QPS (Queries Per Second)**: Throughput αναζητήσεων
- **Speedup**: Επιτάχυνση σε σχέση με exhaustive search

## Παράμετροι Tuning

### Για καλύτερη ποιότητα (υψηλότερο Recall):
- Αύξηση `kclusters`: περισσότερα fine-grained clusters
- Αύξηση `nprobe`: έλεγχος περισσότερων clusters
- Αύξηση `M`: περισσότερη ανάλυση στο PQ
- Αύξηση `nbits`: μεγαλύτερα PQ codebooks

### Για ταχύτερη αναζήτηση:
- Μείωση `nprobe`: λιγότεροι έλεγχοι
- Μείωση `M`: λιγότεροι υπολογισμοί απόστασης
- Μείωση `nbits`: μικρότερα codebooks

## Developers
- Πετρίδου Ελισάβετ - sdi2300170
- Κυριακού Χρήστος - sdi2300096
