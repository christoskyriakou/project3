#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kmeans.h"
#include "ivfflat.h"
#include "dataload.h"

// Usage:
// ./ivfflat_knn <type> <datafile> <k> <nprobe> <outfile>
//   type: "mnist" or "sift"
//   datafile: path to base set
//   k: number of neighbors per point
//   nprobe: number of clusters to probe
//   outfile: binary file with n * k int32 ids

int main(int argc, char** argv)
{
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <mnist|sift> <datafile> <k> <nprobe> <outfile>\n",
            argv[0]);
        return 1;
    }

    const char* type = argv[1];
    const char* datafile = argv[2];
    int k = atoi(argv[3]);
    int nprobe = atoi(argv[4]);
    const char* outfile = argv[5];

    int n = 0;
    int dim = 0;
    Vector** vectors = NULL;

    if (strcmp(type, "mnist") == 0) {
        vectors = load_mnist(datafile, &n, &dim, 0);
    } else if (strcmp(type, "sift") == 0) {
        vectors = load_sift(datafile, &n, &dim, 0);
    } else {
        fprintf(stderr, "Unknown type '%s' (expected mnist or sift)\n", type);
        return 1;
    }

    if (!vectors) {
        fprintf(stderr, "Error: could not load dataset from %s\n", datafile);
        return 1;
    }

    fprintf(stdout, "Loaded %d vectors of dimension %d\n", n, dim);

    int num_clusters = 128;      // μπορείς να το κάνεις παράμετρο αν θες
    unsigned int seed = 1;       // ή να το παίρνεις από argv

    IVFFlat* index = create_ivfflat(num_clusters, dim);
    if (!index) {
        fprintf(stderr, "Error: could not create IVFFlat index\n");
        return 1;
    }

    build_ivfflat(index, vectors, n, seed);

    FILE* fout = fopen(outfile, "wb");
    if (!fout) {
        fprintf(stderr, "Error: could not open outfile %s\n", outfile);
        free_ivfflat(index);
        for (int i = 0; i < n; i++) {
            free_vector(vectors[i]);
        }
        free(vectors);
        return 1;
    }

    // Για κάθε vector, αναζήτηση k+1 για να πετάξουμε τον εαυτό του
    for (int i = 0; i < n; i++) {
        int result_count = 0;
        Neighbor* res = search_ivfflat(
            index,
            vectors,
            vectors[i],   // query = ίδιο vector
            nprobe,
            k + 1,        // ζητάμε k+1 ώστε αν ο πρώτος είναι self να τον πετάξουμε
            &result_count
        );

        if (!res || result_count == 0) {
            // αν δεν βρήκε τπτ, γράφουμε k φορές -1
            int minus1 = -1;
            for (int t = 0; t < k; t++) {
                fwrite(&minus1, sizeof(int), 1, fout);
            }
        } else {
            int written = 0;

            for (int t = 0; t < result_count && written < k; t++) {
                int id = res[t].id;

                // πετάμε τον εαυτό του (στο vectors[i]->id)
                if (id == vectors[i]->id) {
                    continue;
                }

                fwrite(&id, sizeof(int), 1, fout);
                written++;
            }

            // αν δεν φτάσαμε k, συμπληρώνουμε με -1
            int minus1 = -1;
            while (written < k) {
                fwrite(&minus1, sizeof(int), 1, fout);
                written++;
            }
        }

        free(res);

        if (i % 1000 == 0) {
            fprintf(stdout, "Processed %d / %d\n", i, n);
        }
    }

    fclose(fout);

    free_ivfflat(index);
    for (int i = 0; i < n; i++) {
        free_vector(vectors[i]);
    }
    free(vectors);

    fprintf(stdout, "Done. Wrote k-NN graph to %s\n", outfile);

    return 0;
}