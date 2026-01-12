Υλοποίηση αλγορίθμου **IVFFlat (Inverted File Flat)** για προσεγγιστική 
αναζήτηση πλησιέστερων γειτόνων (Approximate Nearest Neighbor Search) σε 
διανυσματικούς χώρους.

### Χαρακτηριστικά:
- K-means clustering για οργάνωση δεδομένων
- Inverted File indexing για γρήγορη αναζήτηση
- Silhouette metric για αξιολόγηση clustering
- Υποστήριξη MNIST και SIFT datasets
- Nearest neighbor search (N-NN)
- Range search (R-radius)

---

##  Αρχεία Έργου

```
ivfflat/
├── ivfflat.c/h       # ivfflat function implementation (create, build, search, free);
├── kmeans.c/h        # kmeans and utility function implementation (euclidean distance, vector ops, kmeans ops)
├── dataloading.c/h   # dataloading function implementation
├── search.c          # Main program
├── Makefile          # Build automation
├── res/              # directory with minst/sift resources
└── README.md         # Τεκμηρίωση (αυτό το αρχείο)
```

##  Εγκατάσταση & Μεταγλώττιση

### Απαιτήσεις:
- GCC compiler (C99 ή νεότερο)
- Linux/Unix σύστημα
- Math library (-lm)

### Μεταγλώττιση:

```bash
# Βασική compilation
make

```
Αυτό δημιουργεί το εκτελέσιμο `search`.


---

## 🚀 Εκτέλεση

### Βασική σύνταξη:
```bash
./search -d <input_file> -q <query_file> -type <mnist|sift> -kclusters <int> -nprobe <int> -N <int> -R <double> -o <output_file> -seed <int> -range <true|false> -ivfflat
```

### Παράμετροι:

| Παράμετρος  | Περιγραφή             | Default      | Παράδειγμα 
|-------------|-----------------------|--------------|------------
| `-d`        | Αρχείο δεδομένων      | -            | `input.dat` 
| `-q`        | Αρχείο queries        | -            | `query.dat` 
| `-type`     | Τύπος dataset         | `mnist`      | `mnist` ή `sift` 
| `-kclusters`| Αριθμός clusters      | `50`         | `50`, `100` 
| `-nprobe`   | Clusters προς έλεγχο  | `5`          | `5`, `10` 
| `-N`        | Πλησιέστεροι γείτονες | `1`          | `5`, `10`, `20` 
| `-R`        | Ακτίνα αναζήτησης     | `2000`/`2`   | MNIST: `2000`, SIFT: `2.0` 
| `-o`        | Αρχείο εξόδου         | `output.txt` | `results.txt` 
| `-seed`     | Random seed           | `1`          | `1`, `42` 
| `-range`    | Range search          | `true`       | `true`, `false` 
| `-ivfflat`  | IVFFlat mode flag     | -            | - 

---

## 💡 Παραδείγματα Χρήσης

### 1. MNIST Dataset (10 πλησιέστεροι):
./search res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte -type mnist -kclusters 50 -nprobe 5 -N 10 -R 2000 -o output.txt -ivfflat

### 2. SIFT Dataset (20 πλησιέστεροι):
./search -d res/sift_base.fvecs -q res/sift_query.fvecs -type sift -kclusters 100 -nprobe 10 -N 20 -R 2.0 -o output.txt -ivfflat

### 3. Χωρίς range search:
./search res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte -type mnist -kclusters 30 -nprobe 3 -N 5 -range false -o output.txt -ivfflat


---

## 📊 Μορφή Εξόδου 

```
Query: 31
Nearest neighbor-1: 5559
distanceApproximate: 109.192490
distanceTrue: 109.192490
Nearest neighbor-2: 4995
distanceApproximate: 126.952744
distanceTrue: 126.952744
Nearest neighbor-3: 4962
distanceApproximate: 175.772583
distanceTrue: 175.772583
R-near neighbors 3:
4962
4995
5559
```

---

## Αλγόριθμος

### 1. Coarse Quantization
- Εφαρμογή **K-means** στο dataset για δημιουργία `kclusters` κεντροειδών (coarse centroids)  
- Δημιουργία **inverted index**: κάθε cluster περιέχει λίστα δεικτών (ή πλήρη διανύσματα) που ανήκουν σε αυτό

### 2. Αποθήκευση Διανυσμάτων
- Κάθε διάνυσμα αποθηκεύεται **σε πλήρη ακρίβεια (float)** μέσα στο αντίστοιχο cluster  
- Δεν γίνεται product quantization ή συμπίεση — τα διανύσματα διατηρούνται “as is”

### 3. Αναζήτηση
- Υπολόγισε τους `nprobe` πλησιέστερους κεντροειδείς ως προς το query  
- Για κάθε επιλεγμένο cluster:
  - Υπολόγισε **ευκλείδειες αποστάσεις** (ή cosine) μεταξύ του query και όλων των διανυσμάτων του cluster  
  - Δεν υπάρχει PQ κωδικοποίηση — οι αποστάσεις είναι **ακριβείς (exact)**  
- Επέστρεψε τα top-N πλησιέστερα διανύσματα από όλους τους εξεταζόμενους clusters

---

## Μετρικές Απόδοσης

- **Recall@N**: Ποσοστό των σωστών (ground truth) αποτελεσμάτων στα top-N  
- **QPS (Queries Per Second)**: Ρυθμός εκτέλεσης αναζητήσεων  
- **Speedup**: Επιτάχυνση σε σχέση με **exhaustive flat search** (εξαρτάται από `nprobe`)  
- **Memory Usage**: Κατανάλωση μνήμης (μεγαλύτερη από IVF-PQ λόγω πλήρων float διανυσμάτων)

---

## Παράμετροι Tuning

### Για καλύτερη ποιότητα (υψηλότερο Recall):
- Αύξηση `kclusters`: περισσότερα λεπτομερή clusters → μικρότερο σφάλμα κβαντοποίησης  
- Αύξηση `nprobe`: έλεγχος περισσότερων clusters κατά την αναζήτηση  
- (Δεν υπάρχουν παράμετροι PQ, καθώς δεν υπάρχει συμπίεση)

### Για ταχύτερη αναζήτηση:
- Μείωση `nprobe`: εξετάζονται λιγότερα clusters  
- Μείωση `kclusters`: λιγότερη υπολογιστική επιβάρυνση στο coarse search  
- Προαιρετικά: χρήση float16 ή SIMD βελτιστοποιήσεων για αποστάσεις

---

## Developers
- Κυριακού Χρήστος - sdi2300096
- Πετρίδου Ελισάβετ - sdi2300170
