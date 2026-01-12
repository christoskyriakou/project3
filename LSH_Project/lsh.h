#ifndef LSH_H
#define LSH_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#define K_MAX 200
#define L_MAX 20
#define DATA_TYPE_MNIST 0
#define DATA_TYPE_SIFT 1
extern int L; 
extern int K;
extern double W;
extern int N;
extern double R;
extern int SEED; 
#define M_HASH_SIZE 69001 

typedef struct {
    int *vectors; // Δυναμικός πίνακας με δείκτες σε διανύσματα
    int count; // Τρέχων αριθμός διανυσμάτων στο bucket
    int capacity; // Μέγιστη χωρητικότητα του bucket
} bucket;

typedef struct {
    bucket *buckets; // Πίνακας με buckets
    double t[K_MAX];// Τυχαίες μετατοπίσεις t_i
    double **r;//  Πίνακας με δείκτες σε τυχαία διανύσματα r
    int v[K_MAX]; // Τυχαίες τιμές v_i για τον υπολογισμό του g
} hashtable;

typedef struct {
    int index; // Δείκτης του διανύσματος
    double distance; //Απόσραση από το query διάνυσμα
} NearestNeighbor;

void initBucket(bucket *b);
void freeBucket(bucket* b);
void resize_bucket(bucket* b);
void addToBucket(bucket *b, int vector);
double scalar_product(const double *vector_v, const double *vector_r, int dim);
int h_funct(const double* vector_v, const double* vector_r, double t, int dim);
int get_g(const double *vector_v, hashtable* h, int dim);
void initHashTable(hashtable* h, int dim);
void insertVector(hashtable* h, const double* vector_v, int vector_index, int dim);
void freeHashTable(hashtable* h);

NearestNeighbor* find_l_nn(
    hashtable lsh_tables[], 
    const double* query_vec, 
    const double* train_dataset, 
    int num_train_vectors, 
    int N, 
    int dim
);

double randn();
double euclidean_distance(const double *v1, const double *v2, int dim);
int compareNeighbors(const void* a, const void* b);

#endif 