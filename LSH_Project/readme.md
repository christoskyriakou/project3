# LSH: Locality Sensitive Hashing Search

## Περιγραφή
Υλοποίηση του αλγορίθμου **LSH (Locality Sensitive Hashing)** για προσεγγιστική 
αναζήτηση πλησιέστερων γειτόνων (Approximate Nearest Neighbors) σε 
υψηλοδιάστατους χώρους, βασισμένη σε προβολές με τυχαίες υπερ-επιφάνειες.

## Χαρακτηριστικά
- Προβολή διανυσμάτων σε τυχαίους προσανατολισμούς (hash functions)
- Κατακερματισμός σε πολλαπλούς πίνακες (L hash tables)
- Γρήγορη εύρεση κοντινών σημείων μέσω hash collisions
- Υποστήριξη **MNIST** και **SIFT** datasets
- **N-nearest neighbors** και **Range (R-near)** αναζήτηση
- Μετρικές απόδοσης: **AF**, **Recall@N**, **QPS**, **tApproximateAverage**, **tTrueAverage**

## Δομή Αρχείων 
LSH_Project/
├── lsh.c             # Υλοποίηση LSH (hash functions, hash tables, search)
├── lsh.h             # Header για lsh.c
├── lsh.o             # Αντικείμενο μετά από compile του lsh.c
├── main.c            # Main πρόγραμμα: parsing arguments, εκτέλεση αναζήτησης
├── main.o            # Αντικείμενο μετά από compile του main.c
├── lsh_app           # Εκτελέσιμο πρόγραμμα που παράγεται από το Makefile
├── Makefile          # Build automation
├── output.txt        # Αρχείο εξόδου αποτελεσμάτων
└── README.md         # Τεκμηρίωση (αυτό το αρχείο)

## Μεταγλώττιση

```bash
make
```
Αυτό δημιουργεί το εκτελέσιμο `search`.

### Παράμετροι
- `-d <file>` - Αρχείο dataset (υποχρεωτικό)
- `-q <file>` - Αρχείο queries (υποχρεωτικό)
- `-o <file>` - Αρχείο εξόδου (default: output.txt)
- `-type <mnist|sift>` - Τύπος dataset (default: mnist)
- `-seed <int>` - Random seed (default: 1)
- `-k <int>` - Πλήθος LSH συναρτήσεων ανά g (default: 4)
- `-L <int>` - Πλήθος πινάκων κατακερματισμού (default: 5)
- `-w <float>` - Μήκος κελιού προβολής (default: 4.0)
- `-N <int>` - Αριθμός πλησιέστερων γειτόνων (default: 1)
- `-R <float>` - Ακτίνα για range search (default: 2000 για MNIST, 2 για SIFT)
- `-range <true|false>` - Ενεργοποίηση range search (default: true)
- `-lsh` - Flag για επιλογή μεθόδου LSH
### Παραδείγματα
#### MNIST
```bash
./search   -d res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte   -k 10 -L 40 -w 200 -N 5 -R 1200   -o results.txt   -type mnist -range true -lsh

```
#### SIFT
```bash
./search  -d res/sift_base.fvecs -q res/sift_query.fvecs   -k 16   -L 1   -w 600   -N 5   -R 30   --o results.txt  -type sift   -range false   -lsh
## Μορφή Εξόδου


```
```
LSH
Query: 0
Nearest neighbor-1: 932085
distanceApproximate: 232.871209
distanceTrue: 232.871209
Nearest neighbor-2: 934876
distanceApproximate: 234.714720
distanceTrue: 234.714720
Nearest neighbor-3: 561813
....
distanceApproximate: 243.989754
distanceTrue: 243.989754
Nearest neighbor-4: 708177
distanceApproximate: 255.460369
distanceTrue: 255.460369
Nearest neighbor-5: 706771
distanceApproximate: 256.314260
distanceTrue: 256.314260
R-near neighbors:
Range Search Disabled
...
Average AF: 1.0272
Recall@5: 0.7780
QPS: 7.41
tApproximateAverage: 0.134912 sec
tTrueAverage: 0.357533 sec
```
#### Αλγόριθμος

### 1. Hash Function Construction
- Δημιουργία `k` τυχαίων διανυσμάτων προβολής `v_i`
- Ορισμός hash συναρτήσεων:  
  `h_i(p) = floor((v_i · p + t_i) / w)` με `t_i ∈ [0, w)`
- Συνένωση `k` hash τιμών σε g-function:  
  `g(p) = (h₁(p), h₂(p), ..., hₖ(p))`

### 2. Κατασκευή Πινάκων (Index Construction)
- Δημιουργία `L` ανεξάρτητων hash tables
- Εισαγωγή κάθε διανύσματος στους κάδους των αντίστοιχων `g` τιμών

### 3. Αναζήτηση (Query Phase)
- Υπολογισμός `L` g-functions για το query
- Ανάκτηση υποψήφιων γειτόνων από τους αντίστοιχους κάδους
- Υπολογισμός ευκλείδειας απόστασης με τους υποψήφιους
- Επιστροφή top-N ή R-near neighbors

## Μετρικές Απόδοσης

- **AF (Approximation Factor)**: Μέσος λόγος απόστασης προσεγγιστικής / ακριβούς
- **Recall@N**: Ποσοστό σωστών αποτελεσμάτων στα top-N
- **QPS (Queries Per Second)**: Throughput αναζητήσεων
- **tApproximateAverage**: Μέσος χρόνος για approximate αναζήτηση
- **tTrueAverage**: Μέσος χρόνος για exact αναζήτηση

## Παράμετροι Tuning

### Για καλύτερη ποιότητα (υψηλότερο Recall):
- Αύξηση `L`: περισσότεροι πίνακες → περισσότερες συγκρούσεις → καλύτερη κάλυψη
- Αύξηση `k`: μεγαλύτερη διακριτότητα hash → πιο ακριβείς συγκρίσεις
- Μείωση `w`: μικρότερα buckets → πιο ακριβής ταξινόμηση

### Για ταχύτερη αναζήτηση:
- Μείωση `L`: λιγότεροι πίνακες → ταχύτερη αναζήτηση
- Μείωση `k`: λιγότερες hash λειτουργίες → ταχύτερο indexing
- Αύξηση `w`: μεγαλύτερα buckets → λιγότερα υποψήφια σημεία

## Developers
- Πετρίδου Ελισάβετ - sdi2300170
- Κυριακού Χρήστος - sdi2300096








