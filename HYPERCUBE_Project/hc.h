#ifndef HC_H
#define HC_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#define K_MAX_HC 20
#define DEFAULT_M 10000 
#define DEFAULT_PROBES 100

typedef struct {
    int *vectors; // Δυναμικός πίνακας με δείκτες σε διανύσματα
    int count; // Τρέχων αριθμός διανυσμάτων στο bucket
    int capacity; // Μέγιστη χωρητικότητα του bucket
} bucket;

typedef struct{
    bucket* buckets; // Πίνακας με buckets
    double **r; //  Πίνακας με δείκτες σε τυχαία διανύσματα r
    double *t;  // Τυχαίες μετατοπίσεις t 
    int kproj; // διαστάσεις προβολής k
    double W; // Το μήκος του κελιού W
}hypercube;

typedef struct {
    int index; // Δείκτης του διανύσματος
    double distance; //Απόσραση από το query διάνυσμα
} NearestNeighbor;

void initBucket(bucket *b);
void resize_bucket(bucket* b);
void addToBucket(bucket *b,int vector);
void freeBucket(bucket* b);
void initHypercube(hypercube* h,int dim,int k,double W);
void FreeHypercube(hypercube* h);
void insertHypercubeVector(hypercube* hc,const double* vector_v,int vector_index,int dim);
int get_g(hypercube* hc,const double* vector,int dim);
int compareNeighbors(const void* a,const void* b);
NearestNeighbor* find_nn_hc(hypercube* h,const double* q,const double* dataset,int num_images,int N,int max_candidates_M,int dim,int probes);
double euclidean_distance(const double *v1,const double *v2,int dim);
double scalar_product(const double *vector_v,const double *vector_r,int dim);
int hamming_dist(int a,int b);
double randn();
#endif 