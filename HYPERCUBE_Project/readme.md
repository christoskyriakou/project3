# Hypercube: Approximate Nearest Neighbor Search

## Περιγραφή
Υλοποιηση του αλγορίθμου **Hypercube** για προσεγγιστική αναζήτηση πλησιέστερων γειτόνων (Approximate Nearest Neighbors) σε υψηλοδιάστατους χώρους, βασισμένη σε τυχαίες προβολές σε υπερκύβους.

## Χαρακτηριστικά
- Τυχαίες προβολές διανυσμάτων σε υπερκύβους
- Εισαγωγή διανυσμάτων σε γωνίες υπερκύβου (vertices)
- Γρήγορη εύρεση κοντινών σημείων μέσω traversal του hypercube
- Υποστήριξη **MNIST** και **SIFT** datasets
- **N-nearest neighbors** και **Range (R-near)** αναζήτηση
- Μετρικές απόδοσης: **AF**, **Recall@N**, **QPS**, **tApproximateAverage**, **tTrueAverage**

## Δομή Αρχείων 
HYPERCUBE_Project/
├── hc.c             # Υλοποίηση Hypercube (hash functions, search)
├── hc.h             # Header για hc.c
├── hc.o             # Αντικείμενο μετά από compile του hc.c
├── main.c           # Main πρόγραμμα (parsing arguments, execution)
├── main.o           # Αντικείμενο μετά από compile του main.c
├── hypercube        # Εκτελέσιμο πρόγραμμα που παράγεται από το Makefile
├── Makefile         # Build automation
└── results.txt      # Αρχείο εξόδου αποτελεσμάτων

## Μεταγλώττιση

```bash
make
```
Αυτό δημιουργεί το εκτελέσιμο `search`.

### Παραδείγματα

#### MNIST
```bash
./search  -d res/train-images.idx3-ubyte -q res/t10k-images.idx3-ubyte  -kproj 14 -w 4.0 -M 1000 -probes 20   -o results.txt -N 5 -R 2.0 -type mnist -range false -hypercube
```

#### SIFT
```bash
./search -d res/sift_base.fvecs -q res/sift_query.fvecs   -kproj 12 -w 15.0 -M 500000 -probes 1000  -o results.txt -N 5 -R 2.0 -type sift -range false -hypercube
```

## Μορφή Εξόδου

```
Hypercube
Query: 0
Nearest neighbor-1: image_id_882
distanceApproximate: 0.555305
distanceTrue: 0.458501
Nearest neighbor-2: image_id_190
distanceApproximate: 0.575745
....
Nearest neighbor-3: image_id_816
distanceApproximate: 0.611712
distanceTrue: 0.480346
Nearest neighbor-4: image_id_224
distanceApproximate: 0.617551
distanceTrue: 0.503015
Nearest neighbor-5: image_id_292
distanceApproximate: 0.618363
distanceTrue: 0.504624
...
Average AF: 1.003456
Recall@N: 0.987654
QPS: 123.45
tApproximateAverage: 0.008100
tTrueAverage: 1.234567
```
## Αλγόριθμος

### 1. Projection & Hash Functions
- Δημιουργία `kproj` τυχαίων διανυσμάτων προβολής
- Ορισμός hash συναρτήσεων για κάθε διάσταση προβολής
- Ανάθεση κάθε διανύσματος σε vertex του hypercube βάσει hash τιμών

### 2. Κατασκευή Hypercube
- Δημιουργία υπερκύβου με 2^kproj κορυφές
- Εισαγωγή διανυσμάτων στους αντίστοιχους vertices

### 3. Αναζήτηση (Query Phase)
- Υπολογισμός hash functions για το query
- Traversal μέχρι `probes` κορυφές
- Έλεγχος μέχρι `M` υποψήφιων σημείων
- Επιστροφή top-N ή R-near neighbors

## Μετρικές Απόδοσης
- **AF (Approximation Factor)**: Μέσος λόγος απόστασης προσεγγιστικής / ακριβούς
- **Recall@N**: Ποσοστό σωστών αποτελεσμάτων στα top-N
- **QPS (Queries Per Second)**: Throughput αναζητήσεων
- **tApproximateAverage**: Μέσος χρόνος για approximate αναζήτηση
- **tTrueAverage**: Μέσος χρόνος για exact αναζήτηση

## Παράμετροι Tuning

### Για καλύτερη ποιότητα (υψηλότερο Recall):
- Αύξηση `kproj`: περισσότερη διακριτότητα hash → ακριβέστερα vertices
- Αύξηση `probes`: έλεγχος περισσότερων κορυφών → καλύτερη κάλυψη
- Μείωση `w`: μικρότερα cells → πιο ακριβής ταξινόμηση

### Για ταχύτερη αναζήτηση:
- Μείωση `kproj`: λιγότερες hash λειτουργίες → ταχύτερο indexing
- Μείωση `probes`: λιγότερα traversal → ταχύτερη αναζήτηση
- Αύξηση `w`: μεγαλύτερα cells → λιγότερα υποψήφια σημεία

## Developers
- Πετρίδου Ελισάβετ - sdi2300170
- Κυριακού Χρήστος - sdi2300096













