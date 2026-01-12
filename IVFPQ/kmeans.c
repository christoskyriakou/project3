#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include "dataload.h"

KMeans* kmeans_create(int k, int dim) {
    KMeans *km = (KMeans*)malloc(sizeof(KMeans));
    km->k = k;
    km->dim = dim;
    km->centroids = (Vector*)malloc(k * sizeof(Vector));
    km->clusters = (Cluster*)malloc(k * sizeof(Cluster));
    
    for (int i = 0; i < k; i++) {
        km->centroids[i].coords = (float*)calloc(dim, sizeof(float));
        km->centroids[i].dim = dim;
        km->centroids[i].id = i;
        km->clusters[i].assignments = NULL;
        km->clusters[i].size =  0;
    } 
    
    return km;
}

void kmeans_init_centroids(KMeans *km, Vector *data, int n, unsigned int seed) {
    if (n < km->k) {
        printf("Warning: n=%d < k=%d\n", n, km->k);
        for (int k = 0; k < km->k && k < n; k++) {
            memcpy(km->centroids[k].coords, data[k].coords, km->dim * sizeof(float));
        }
        return;
    }
    
    // K-means++ initialization
    srand(seed);
    int first = rand() % n;
    memcpy(km->centroids[0].coords, data[first].coords, km->dim * sizeof(float));
    
    for (int k = 1; k < km->k; k++) {
        float *distances = (float*)malloc(n * sizeof(float));
        float sum = 0.0;
        
        for (int i = 0; i < n; i++) {
            float min_dist = FLT_MAX;
            for (int j = 0; j < k; j++) {
                float dist = euclidean_distance(data[i].coords, km->centroids[j].coords, km->dim);
                if (dist < min_dist) min_dist = dist;
            }
            distances[i] = min_dist * min_dist;
            sum += distances[i];
        }
        
        if (sum < 1e-10) {
            // All remaining points are duplicates, just pick randomly
            int selected = rand_int(n);
            memcpy(km->centroids[k].coords, data[selected].coords, km->dim * sizeof(float));
        } else {
            float r = ((float)rand() / RAND_MAX) * sum;
            float cumsum = 0.0;
            int selected = n - 1;
            
            for (int i = 0; i < n; i++) {
                cumsum += distances[i];
                if (cumsum >= r) {
                    selected = i;
                    break;
                }
            }
            
            memcpy(km->centroids[k].coords, data[selected].coords, km->dim * sizeof(float));
        }
        
        free(distances);
    }
}

void kmeans_train(KMeans *km, Vector *data, int n, int max_iter) {
    int *assignments = (int*)malloc(n * sizeof(int));
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Assignment step
        int changed = 0;
        for (int i = 0; i < n; i++) {
            float min_dist = FLT_MAX;
            int best_cluster = 0;
            
            for (int k = 0; k < km->k; k++) {
                float dist = euclidean_distance(data[i].coords, km->centroids[k].coords, km->dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            if (assignments[i] != best_cluster) changed = 1;
            assignments[i] = best_cluster;
        }
        
        if (!changed && iter > 0) break;
        
        // Update step
        int *counts = (int*)calloc(km->k, sizeof(int));
        for (int k = 0; k < km->k; k++) {
            memset(km->centroids[k].coords, 0, km->dim * sizeof(float));
        }
        
        for (int i = 0; i < n; i++) {
            int cluster = assignments[i];
            for (int d = 0; d < km->dim; d++) {
                km->centroids[cluster].coords[d] += data[i].coords[d];
            }
            counts[cluster]++;
        }
        
        for (int k = 0; k < km->k; k++) {
            if (counts[k] > 0) {
                for (int d = 0; d < km->dim; d++) {
                    km->centroids[k].coords[d] /= counts[k];
                }
            }
        }
        
        free(counts);
    }
    
    // Store final assignments
    for (int k = 0; k < km->k; k++) {
        km->clusters[k].size = 0;
        for (int i = 0; i < n; i++) {
            if (assignments[i] == k) km->clusters[k].size++;
        }
        km->clusters[k].assignments = (int*)malloc(km->clusters[k].size * sizeof(int));
        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (assignments[i] == k) {
                km->clusters[k].assignments[idx++] = i;
            }
        }
    }
    
    free(assignments);
}

void kmeans_free(KMeans *km) {
    for (int i = 0; i < km->k; i++) {
        free(km->centroids[i].coords);
        free(km->clusters[i].assignments);
    }
    free(km->centroids);
    free(km->clusters);
    free(km);
}


// Silhouette 

float compute_silhouette_score(Vector *data, int n, int *assignments, Vector *centroids, int k, int dim) {
    float total_silhouette = 0.0;
    int valid_points = 0;
    
    for (int i = 0; i < n; i++) {
        int cluster_i = assignments[i];
        
        // Compute a(i): average distance to points in same cluster
        float a_i = 0.0;
        int same_cluster_count = 0;
        
        for (int j = 0; j < n; j++) {
            if (i != j && assignments[j] == cluster_i) {
                a_i += euclidean_distance(data[i].coords, data[j].coords, dim);
                same_cluster_count++;
            }
        }
        
        if (same_cluster_count > 0) {
            a_i /= same_cluster_count;
        } else {
            // Point is alone in cluster, silhouette = 0
            continue;
        }
        
        // Compute b(i): minimum average distance to points in other clusters
        float b_i = FLT_MAX;
        
        for (int c = 0; c < k; c++) {
            if (c == cluster_i) continue;
            
            float avg_dist = 0.0;
            int other_cluster_count = 0;
            
            for (int j = 0; j < n; j++) {
                if (assignments[j] == c) {
                    avg_dist += euclidean_distance(data[i].coords, data[j].coords, dim);
                    other_cluster_count++;
                }
            }
            
            if (other_cluster_count > 0) {
                avg_dist /= other_cluster_count;
                if (avg_dist < b_i) {
                    b_i = avg_dist;
                }
            }
        }
        
        // Compute silhouette for this point
        if (b_i != FLT_MAX) {
            float s_i = (b_i - a_i) / fmaxf(a_i, b_i);
            total_silhouette += s_i;
            valid_points++;
        }
    }
    
    return valid_points > 0 ? total_silhouette / valid_points : -1.0;
}

float compute_silhouette_sample(Vector *data, int n, int *assignments, Vector *centroids, int k, int dim, int sample_size) {
    // For large datasets, compute silhouette on a sample
    if (n <= sample_size) {
        return compute_silhouette_score(data, n, assignments, centroids, k, dim);
    }
    
    // Random sampling
    int *sampled_indices = (int*)malloc(sample_size * sizeof(int));
    int *used = (int*)calloc(n, sizeof(int));
    
    for (int i = 0; i < sample_size; i++) {
        int idx;
        do {
            idx = rand_int(n);
        } while (used[idx]);
        used[idx] = 1;
        sampled_indices[i] = idx;
    }
    
    free(used);
    
    // Create sampled data
    Vector *sampled_data = (Vector*)malloc(sample_size * sizeof(Vector));
    int *sampled_assignments = (int*)malloc(sample_size * sizeof(int));
    
    for (int i = 0; i < sample_size; i++) {
        int idx = sampled_indices[i];
        sampled_data[i].coords = data[idx].coords;
        sampled_data[i].dim = data[idx].dim;
        sampled_data[i].id = data[idx].id;
        sampled_assignments[i] = assignments[idx];
    }
    
    float score = compute_silhouette_score(sampled_data, sample_size, sampled_assignments, centroids, k, dim);
    
    free(sampled_data);
    free(sampled_assignments);
    free(sampled_indices);
    
    return score;
}

int find_optimal_k_silhouette(Vector *data, int n, int dim, int k_min, int k_max, int sample_size, unsigned int seed) {
    printf("\n=== Finding optimal k using Silhouette Score ===\n");
    printf("Testing k from %d to %d\n", k_min, k_max);
    printf("Sample size: %d\n\n", sample_size);
    
    float best_score = -2.0;
    int best_k = k_min;
    
    for (int k = k_min; k <= k_max; k++) {
        printf("Testing k=%d... ", k);
        fflush(stdout);
        
        // Create and train k-means
        KMeans *km = kmeans_create(k, dim);
        kmeans_init_centroids(km, data, n, seed);
        kmeans_train(km, data, n, 20);
        
        // Get assignments
        int *assignments = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            float min_dist = FLT_MAX;
            int best_cluster = 0;
            
            for (int c = 0; c < k; c++) {
                float dist = euclidean_distance(data[i].coords, km->centroids[c].coords, dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }
        
        // Compute silhouette score
        float score = compute_silhouette_sample(data, n, assignments, km->centroids, k, dim, sample_size);
        
        printf("Silhouette Score = %.4f", score);
        
        if (score > best_score) {
            best_score = score;
            best_k = k;
            printf(" (NEW BEST!)");
        }
        printf("\n");
        
        free(assignments);
        kmeans_free(km);
    }
    
    printf("\n=== Optimal k = %d with Silhouette Score = %.4f ===\n\n", best_k, best_score);
    
    return best_k;
}
