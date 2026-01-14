#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include "dataload.h"


// Urility functions

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float euclidean_distance(float *a, float *b, int dim) {
    float sum = 0.0;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

float asymmetric_distance(float *query_subvec, Vector *centroids, unsigned char code, int subvec_dim) {
    return euclidean_distance(query_subvec, centroids[code].coords, subvec_dim);
}

void init_random(unsigned int seed) {
    srand(seed);
}

int rand_int(int max) {
    return rand() % max;
}

int compare_results(const void *a, const void *b) {
    float diff = ((SearchResult*)a)->distance - ((SearchResult*)b)->distance;
    return (diff > 0) - (diff < 0);
}

float compute_recall(SearchResult *approx, int approx_count, SearchResult *exact, int exact_count) {
    if (exact_count == 0) return 0.0;
    
    int matches = 0;
    for (int i = 0; i < approx_count; i++) {
        for (int j = 0; j < exact_count; j++) {
            if (approx[i].id == exact[j].id) {
                matches++;
                break;
            }
        }
    }
    
    return (float)matches / exact_count;
}

// Data Load

Vector* load_mnist(const char *filename, int *count, int *dim, int isQuery) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    int t = 0;
    t++;
    
    unsigned int magic, n_images, rows, cols;
    t = fread(&magic, 4, 1, fp);
    t = fread(&n_images, 4, 1, fp);
    t = fread(&rows, 4, 1, fp);
    t = fread(&cols, 4, 1, fp);
    
    magic = __builtin_bswap32(magic);
    n_images = __builtin_bswap32(n_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);
    
    *count = n_images;
    *dim = rows * cols;
    if (isQuery) {
        *count = 500;
    } else {
        *count = 5000;
    }
    
    
    Vector *vectors = (Vector*)malloc(n_images * sizeof(Vector));
    
    for (int i = 0; i < n_images; i++) {
        vectors[i].dim = *dim;
        vectors[i].id = i;
        vectors[i].coords = (float*)malloc(*dim * sizeof(float));
        
        unsigned char *buffer = (unsigned char*)malloc(*dim);
        t = fread(buffer, 1, *dim, fp);
        
        for (int j = 0; j < *dim; j++) {
            vectors[i].coords[j] = (float)buffer[j];
        }
        free(buffer);
    }
    
    fclose(fp);
    printf("Loaded %d MNIST vectors (dim=%d)\n", *count, *dim);
    return vectors;
}

Vector* load_sift(const char *filename, int *count, int *dim, int isQuery) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    int t = 0;
    t++;

    int first_dim;
    t = fread(&first_dim, 4, 1, fp);
    fseek(fp, 0, SEEK_SET);
    
    *dim = first_dim;
    int bytes_per_vector = 4 + first_dim * 4; 
    *count = file_size / bytes_per_vector;
    if (isQuery) {
        *count = 1000;
    } else {
        *count = 10000;
    }
    
    Vector *vectors = (Vector*)malloc(*count * sizeof(Vector));
    
    for (int i = 0; i < *count; i++) {
        int d;
        t = fread(&d, 4, 1, fp);
        
        vectors[i].dim = d;
        vectors[i].id = i;
        vectors[i].coords = (float*)malloc(d * sizeof(float));
        t = fread(vectors[i].coords, sizeof(float), d, fp);
    }
    
    fclose(fp);
    printf("Loaded %d SIFT vectors (dim=%d)\n", *count, *dim);
    return vectors;
}

void free_vectors(Vector *vectors, int count) {
    for (int i = 0; i < count; i++) {
        free(vectors[i].coords);
    }
    free(vectors);
}