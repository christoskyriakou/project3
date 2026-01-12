#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include "dataload.h"
#include "kmeans.h"


IVFPQ* ivfpq_create(int kclusters, int M, int nbits, int dim) {
    IVFPQ *index = (IVFPQ*)malloc(sizeof(IVFPQ));
    index->kclusters = kclusters;
    index->M = M;
    index->nbits = nbits;
    index->dim = dim;
    index->subvec_dim = dim / M;
    
    if (dim % M != 0) {
        printf("Warning: dim=%d not divisible by M=%d, using subvec_dim=%d\n", dim, M, index->subvec_dim);
    }
    
    index->coarse_quantizer = kmeans_create(kclusters, dim);
    index->pq_centroids = (Vector**)malloc(M * sizeof(Vector*));
    
    int k_pq = 1 << nbits;
    for (int m = 0; m < M; m++) {
        index->pq_centroids[m] = (Vector*)malloc(k_pq * sizeof(Vector));
        for (int k = 0; k < k_pq; k++) {
            index->pq_centroids[m][k].coords = (float*)calloc(index->subvec_dim, sizeof(float));
            index->pq_centroids[m][k].dim = index->subvec_dim;
            index->pq_centroids[m][k].id = k;
        }
    }
    
    index->inverted_lists = (InvertedList*)malloc(kclusters * sizeof(InvertedList));
    for (int i = 0; i < kclusters; i++) {
        index->inverted_lists[i].codes = NULL;
        index->inverted_lists[i].vector_ids = NULL;
        index->inverted_lists[i].count = 0;
    }
    
    return index;
}

void ivfpq_train(IVFPQ *index, Vector *data, int n, unsigned int seed) {
    printf("Training coarse quantizer with %d clusters...\n", index->kclusters);
    kmeans_init_centroids(index->coarse_quantizer, data, n, seed);
    kmeans_train(index->coarse_quantizer, data, n, 20);
    
    printf("Training product quantizers (%d subspaces)...\n", index->M);
    int k_pq = 1 << index->nbits;
    
    for (int m = 0; m < index->M; m++) {
        if ((m + 1) % 4 == 0 || m == index->M - 1) {
            printf("  Progress: %d/%d\n", m + 1, index->M);
        }
        
        Vector *subvectors = (Vector*)malloc(n * sizeof(Vector));
        for (int i = 0; i < n; i++) {
            subvectors[i].coords = (float*)malloc(index->subvec_dim * sizeof(float));
            subvectors[i].dim = index->subvec_dim;
            subvectors[i].id = i;
            
            int start_idx = m * index->subvec_dim;
            int end_idx = start_idx + index->subvec_dim;
            if (end_idx > data[i].dim) end_idx = data[i].dim;
            int copy_dim = end_idx - start_idx;
            
            memcpy(subvectors[i].coords, &data[i].coords[start_idx], copy_dim * sizeof(float));
            if (copy_dim < index->subvec_dim) {
                memset(&subvectors[i].coords[copy_dim], 0, 
                      (index->subvec_dim - copy_dim) * sizeof(float));
            }
        }
        
        KMeans *pq_km = kmeans_create(k_pq, index->subvec_dim);
        kmeans_init_centroids(pq_km, subvectors, n, seed);
        kmeans_train(pq_km, subvectors, n, 20);
        
        for (int k = 0; k < k_pq; k++) {
            memcpy(index->pq_centroids[m][k].coords, 
                   pq_km->centroids[k].coords, 
                   index->subvec_dim * sizeof(float));
        }
        
        for (int i = 0; i < n; i++) {
            free(subvectors[i].coords);
        }
        free(subvectors);
        kmeans_free(pq_km);
    }
}

void ivfpq_add(IVFPQ *index, Vector *data, int n) {
    printf("Encoding and adding %d vectors...\n", n);
    
    for (int i = 0; i < n; i++) {
        if ((i + 1) % 10000 == 0) {
            printf("  Encoded: %d/%d\n", i + 1, n);
        }
        
        // Find nearest coarse cluster
        float min_dist = FLT_MAX;
        int best_cluster = 0;
        
        for (int k = 0; k < index->kclusters; k++) {
            float dist = euclidean_distance(data[i].coords, 
                                           index->coarse_quantizer->centroids[k].coords, 
                                           index->dim);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        
        // Encode with PQ
        unsigned char *code = (unsigned char*)malloc(index->M * sizeof(unsigned char));
        
        for (int m = 0; m < index->M; m++) {
            int start_idx = m * index->subvec_dim;
            float *subvec = &data[i].coords[start_idx];
            float min_subdist = FLT_MAX;
            unsigned char best_code = 0;
            
            int k_pq = 1 << index->nbits;
            for (int k = 0; k < k_pq; k++) {
                float dist = euclidean_distance(subvec, 
                                               index->pq_centroids[m][k].coords, 
                                               index->subvec_dim);
                if (dist < min_subdist) {
                    min_subdist = dist;
                    best_code = (unsigned char)k;
                }
            }
            code[m] = best_code;
        }
        
        // Add to inverted list
        InvertedList *list = &index->inverted_lists[best_cluster];
        list->codes = (unsigned char*)realloc(list->codes, (list->count + 1) * index->M * sizeof(unsigned char));
        list->vector_ids = (int*)realloc(list->vector_ids, (list->count + 1) * sizeof(int));
        
        memcpy(&list->codes[list->count * index->M], code, index->M * sizeof(unsigned char));
        list->vector_ids[list->count] = data[i].id;
        list->count++;
        
        free(code);
    }
}

SearchResult* ivfpq_search(IVFPQ *index, Vector *query, int nprobe, int N, int *result_count) {
    // Find nprobe nearest clusters
    SearchResult *cluster_dists = (SearchResult*)malloc(index->kclusters * sizeof(SearchResult));
    
    for (int k = 0; k < index->kclusters; k++) {
        cluster_dists[k].id = k;
        cluster_dists[k].distance = euclidean_distance(query->coords, index->coarse_quantizer->centroids[k].coords, index->dim);
    }
    
    qsort(cluster_dists, index->kclusters, sizeof(SearchResult), compare_results);
    
    // Search in selected clusters
    int max_candidates = 100000;
    SearchResult *candidates = (SearchResult*)malloc(max_candidates * sizeof(SearchResult));
    int cand_count = 0;
    
    for (int p = 0; p < nprobe && p < index->kclusters; p++) {
        int cluster_id = cluster_dists[p].id;
        InvertedList *list = &index->inverted_lists[cluster_id];
        
        for (int i = 0; i < list->count; i++) {
            if (cand_count >= max_candidates) break;
            
            float dist = 0.0;
            
            for (int m = 0; m < index->M; m++) {
                unsigned char code = list->codes[i * index->M + m];
                int start_idx = m * index->subvec_dim;
                float *query_subvec = &query->coords[start_idx];
                float subdist = asymmetric_distance(query_subvec, index->pq_centroids[m], code, index->subvec_dim);
                dist += subdist * subdist;
            }
            
            candidates[cand_count].id = list->vector_ids[i];
            candidates[cand_count].distance = sqrtf(dist);
            cand_count++;
        }
    }
    
    free(cluster_dists);
    
    if (cand_count == 0) {
        *result_count = 0;
        free(candidates);
        return NULL;
    }
    
    qsort(candidates, cand_count, sizeof(SearchResult), compare_results);
    
    int final_count = (N < cand_count) ? N : cand_count;
    SearchResult *results = (SearchResult*)malloc(final_count * sizeof(SearchResult));
    memcpy(results, candidates, final_count * sizeof(SearchResult));
    
    free(candidates);
    *result_count = final_count;
    return results;
}

SearchResult* ivfpq_range_search(IVFPQ *index, Vector *query, float radius, int *result_count) {
    SearchResult *cluster_dists = (SearchResult*)malloc(index->kclusters * sizeof(SearchResult));
    
    for (int k = 0; k < index->kclusters; k++) {
        cluster_dists[k].id = k;
        cluster_dists[k].distance = euclidean_distance(query->coords, index->coarse_quantizer->centroids[k].coords, index->dim);
    }
    
    qsort(cluster_dists, index->kclusters, sizeof(SearchResult), compare_results);
    
    int max_results = 10000;
    SearchResult *results = (SearchResult*)malloc(max_results * sizeof(SearchResult));
    int res_count = 0;
    
    // Check clusters within expanded radius
    for (int p = 0; p < index->kclusters; p++) {
        int cluster_id = cluster_dists[p].id;
        if (cluster_dists[p].distance > radius * 2.0) break;
        
        InvertedList *list = &index->inverted_lists[cluster_id];
        
        for (int i = 0; i < list->count; i++) {
            float dist = 0.0;
            
            for (int m = 0; m < index->M; m++) {
                unsigned char code = list->codes[i * index->M + m];
                int start_idx = m * index->subvec_dim;
                float *query_subvec = &query->coords[start_idx];
                float subdist = asymmetric_distance(query_subvec, index->pq_centroids[m], code, index->subvec_dim);
                dist += subdist * subdist;
            }
            
            dist = sqrtf(dist);
            
            if (dist <= radius) {
                if (res_count >= max_results) {
                    max_results *= 2;
                    results = (SearchResult*)realloc(results, max_results * sizeof(SearchResult));
                }
                results[res_count].id = list->vector_ids[i];
                results[res_count].distance = dist;
                res_count++;
            }
        }
    }
    
    free(cluster_dists);
    *result_count = res_count;
    return results;
}

void ivfpq_free(IVFPQ *index) {
    kmeans_free(index->coarse_quantizer);
    
    int k_pq = 1 << index->nbits;
    for (int m = 0; m < index->M; m++) {
        for (int k = 0; k < k_pq; k++) {
            free(index->pq_centroids[m][k].coords);
        }
        free(index->pq_centroids[m]);
    }
    free(index->pq_centroids);
    
    for (int i = 0; i < index->kclusters; i++) {
        free(index->inverted_lists[i].codes);
        free(index->inverted_lists[i].vector_ids);
    }
    free(index->inverted_lists);
    free(index);
}
