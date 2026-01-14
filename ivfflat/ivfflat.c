#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "kmeans.h"

IVFFlat* create_ivfflat(int num_clusters, int dimension) {
    IVFFlat* index = (IVFFlat*)malloc(sizeof(IVFFlat));
    index->num_clusters = num_clusters;
    index->dimension = dimension; 
    
    index->clusters = (Cluster*)calloc(num_clusters, sizeof(Cluster));
    
    // Initialize clusters
    for (int i = 0; i < num_clusters; i++) {
        index->clusters[i].centroid = create_vector(dimension, i);
        index->clusters[i].members = NULL;
        index->clusters[i].member_count = 0;
        index->clusters[i].capacity = 0;
    }
    
    return index;
}

void build_ivfflat(IVFFlat* index, Vector** vectors, int n, unsigned int seed) {
    printf("Building IVFFlat index with %d clusters...\n", index->num_clusters);

    // Τρέχουμε k-means για να βρούμε τα centroids & τα μέλη
    kmeans(vectors, n, index->clusters, index->num_clusters, index->dimension, 100, seed);

    // Αφαιρέσαμε τον υπολογισμό silhouette γιατί είναι O(N^2) και πολύ αργός για N=10000
    // float silhouette = calculate_silhouette(vectors, n, index->clusters, index->num_clusters);
    // printf("Silhouette score: %.4f\n", silhouette);

    printf("IVFFlat build completed.\n");
}

int compare_neighbors(const void* a, const void* b) {
    Neighbor* n1 = (Neighbor*)a;
    Neighbor* n2 = (Neighbor*)b;
    if (n1->distance < n2->distance) return -1;
    if (n1->distance > n2->distance) return 1;
    return 0;
}

Neighbor* search_ivfflat(IVFFlat* index, Vector** vectors, Vector* query, int nprobe, int N, int* result_count) {
    // Find nprobe nearest clusters
    Neighbor* cluster_distances = (Neighbor*)malloc(index->num_clusters * sizeof(Neighbor));
    
    for (int c = 0; c < index->num_clusters; c++) {
        cluster_distances[c].id = c;
        cluster_distances[c].distance = euclidean_distance(query, index->clusters[c].centroid);
    }
    
    qsort(cluster_distances, index->num_clusters, sizeof(Neighbor), compare_neighbors);
    
    // Search in nprobe nearest clusters
    int max_candidates = 10000;
    Neighbor* candidates = (Neighbor*)malloc(max_candidates * sizeof(Neighbor));
    int candidate_count = 0;
    
    for (int p = 0; p < nprobe && p < index->num_clusters; p++) {
        int cluster_id = cluster_distances[p].id;
        Cluster* cluster = &index->clusters[cluster_id];
        
        for (int m = 0; m < cluster->member_count; m++) {
            int vec_id = cluster->members[m];
            float dist = euclidean_distance(query, vectors[vec_id]);
            
            if (candidate_count < max_candidates) {
                candidates[candidate_count].id = vec_id;
                candidates[candidate_count].distance = dist;
                candidate_count++;
            }
        }
    }
    
    // Sort candidates and return top N
    qsort(candidates, candidate_count, sizeof(Neighbor), compare_neighbors);
    
    *result_count = (N < candidate_count) ? N : candidate_count;
    Neighbor* results = (Neighbor*)malloc(*result_count * sizeof(Neighbor));
    memcpy(results, candidates, *result_count * sizeof(Neighbor));
    
    free(cluster_distances);
    free(candidates);
    
    return results;
}

Neighbor* range_search_ivfflat(IVFFlat* index, Vector** vectors, Vector* query, float radius, int* result_count) {
    int max_results = 1000;
    Neighbor* results = (Neighbor*)malloc(max_results * sizeof(Neighbor));
    *result_count = 0;

    // OPTIMIZATION: Find distances to all cluster centroids
    Neighbor* cluster_distances = (Neighbor*)malloc(index->num_clusters * sizeof(Neighbor));
    
    for (int c = 0; c < index->num_clusters; c++) {
        cluster_distances[c].id = c;
        cluster_distances[c].distance = euclidean_distance(query, index->clusters[c].centroid);
    }
    
    // Sort clusters by distance to query
    qsort(cluster_distances, index->num_clusters, sizeof(Neighbor), compare_neighbors);

    
    for (int i = 0; i < index->num_clusters; i++) {
        int cluster_id = cluster_distances[i].id;
        float centroid_dist = cluster_distances[i].distance;
        
        // Early termination: if centroid is too far, skip remaining clusters, comment to search all clusters
        if (centroid_dist > radius * 2.0f) {
            break;
        }
        Cluster* cluster = &index->clusters[cluster_id];
        
        for (int m = 0; m < cluster->member_count; m++) {
            int vec_id = cluster->members[m];
            float dist = euclidean_distance(query, vectors[vec_id]);
            
            if (dist <= radius) {
                if (*result_count >= max_results) {
                    max_results *= 2;
                    results = (Neighbor*)realloc(results, max_results * sizeof(Neighbor));
                }
                results[*result_count].id = vec_id;
                results[*result_count].distance = dist;
                (*result_count)++;
            }
        }
    }
    
    return results;
}

void free_ivfflat(IVFFlat* index) {
    if (index) {
        for (int i = 0; i < index->num_clusters; i++) {
            free_vector(index->clusters[i].centroid);
            free(index->clusters[i].members);
        }
        free(index->clusters);
        free(index);
    }
}