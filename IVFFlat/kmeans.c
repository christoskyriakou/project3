
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "kmeans.h"


// Vector Operations

// Calculate Euclidean distance (L2)
float euclidean_distance(const Vector* v1, const Vector* v2) {
    float sum = 0.0f;
    for (int i = 0; i < v1->dimension; i++) {
        float diff = v1->coords[i] - v2->coords[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

Vector* create_vector(int dimension, int id) {
    Vector* v = (Vector*)malloc(sizeof(Vector));
    v->coords = (float*)calloc(dimension, sizeof(float));
    v->dimension = dimension;
    v->id = id;
    return v;
}

void copy_vector(Vector* dest, const Vector* src) {
    memcpy(dest->coords, src->coords, src->dimension * sizeof(float));
}

void free_vector(Vector* v) {
    if (v) {
        free(v->coords);
        free(v);
    }
}

// K-means Clustering

// Initialize centroids using k-means++
void initialize_centroids_kmeans(Vector** vectors, int n, Cluster* clusters, int k, int dimension, unsigned int seed) {
    srand(seed);
    
    // Choose first centroid randomly
    int first_idx = rand() % n;
    copy_vector(clusters[0].centroid, vectors[first_idx]);
    
    float* min_distances = (float*)malloc(n * sizeof(float));
    
    // Choose remaining k-1 centroids
    for (int c = 1; c < k; c++) {
        // Calculate distance to nearest centroid for each point
        float total_distance = 0.0f;
        
        for (int i = 0; i < n; i++) {
            float min_dist = FLT_MAX;
            for (int j = 0; j < c; j++) {
                float dist = euclidean_distance(vectors[i], clusters[j].centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            min_distances[i] = min_dist * min_dist; // Squared distance
            total_distance += min_distances[i];
        }
        
        // Choose next centroid with probability proportional to distance
        float rand_val = ((float)rand() / RAND_MAX) * total_distance;
        float cumsum = 0.0f;
        int next_idx = 0;
        
        for (int i = 0; i < n; i++) {
            cumsum += min_distances[i];
            if (cumsum >= rand_val) {
                next_idx = i;
                break;
            }
        }
        
        copy_vector(clusters[c].centroid, vectors[next_idx]);
    }
    
    free(min_distances);
}

int assign_to_clusters(Vector** vectors, int n, Cluster* clusters, int k) {
    int changes = 0;
    
    for (int i = 0; i < n; i++) {
        float min_dist = FLT_MAX;
        int best_cluster = 0;
        
        // Find nearest centroid
        for (int c = 0; c < k; c++) {
            float dist = euclidean_distance(vectors[i], clusters[c].centroid);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        
        // Add to cluster if capacity allows
        Cluster* cluster = &clusters[best_cluster];
        if (cluster->member_count >= cluster->capacity) {
            cluster->capacity = cluster->capacity == 0 ? 100 : cluster->capacity * 2;
            cluster->members = (int*)realloc(cluster->members, cluster->capacity * sizeof(int));
        }
        
        cluster->members[cluster->member_count] = vectors[i]->id;
        cluster->member_count++;
        changes++;
    }
    
    return changes;
}

void update_centroids(Vector** vectors, Cluster* clusters, int k, int dimension) {
    for (int c = 0; c < k; c++) {
        if (clusters[c].member_count == 0) continue;
        
        // Reset centroid
        for (int d = 0; d < dimension; d++) {
            clusters[c].centroid->coords[d] = 0.0f;
        }
        
        // Sum all member vectors
        for (int m = 0; m < clusters[c].member_count; m++) {
            int vec_id = clusters[c].members[m];
            for (int d = 0; d < dimension; d++) {
                clusters[c].centroid->coords[d] += vectors[vec_id]->coords[d];
            }
        }
        
        // Average
        for (int d = 0; d < dimension; d++) {
            clusters[c].centroid->coords[d] /= clusters[c].member_count;
        }
    }
}

float calculate_silhouette(Vector** vectors, int n, Cluster* clusters, int k) {
    float total_silhouette = 0.0f;
    int valid_points = 0;
    
    for (int i = 0; i < n; i++) {
        // Find cluster of current vector
        int my_cluster = -1;
        for (int c = 0; c < k; c++) {
            for (int m = 0; m < clusters[c].member_count; m++) {
                if (clusters[c].members[m] == vectors[i]->id) {
                    my_cluster = c;
                    break;
                }
            }
            if (my_cluster != -1) break;
        }
        
        if (my_cluster == -1 || clusters[my_cluster].member_count <= 1) continue;
        
        // Calculate a(i): average distance to points in same cluster
        float a_i = 0.0f;
        for (int m = 0; m < clusters[my_cluster].member_count; m++) {
            int other_id = clusters[my_cluster].members[m];
            if (other_id != vectors[i]->id) {
                a_i += euclidean_distance(vectors[i], vectors[other_id]);
            }
        }
        a_i /= (clusters[my_cluster].member_count - 1);
        
        // Calculate b(i): min average distance to points in other clusters
        float b_i = FLT_MAX;
        for (int c = 0; c < k; c++) {
            if (c == my_cluster || clusters[c].member_count == 0) continue;
            
            float avg_dist = 0.0f;
            for (int m = 0; m < clusters[c].member_count; m++) {
                int other_id = clusters[c].members[m];
                avg_dist += euclidean_distance(vectors[i], vectors[other_id]);
            }
            avg_dist /= clusters[c].member_count;
            
            if (avg_dist < b_i) {
                b_i = avg_dist;
            }
        }
        
        // Silhouette coefficient for this point
        float s_i = (b_i - a_i) / fmaxf(a_i, b_i);
        total_silhouette += s_i;
        valid_points++;
    }
    
    return valid_points > 0 ? total_silhouette / valid_points : 0.0f;
}

// Αντικαταστήστε την υπάρχουσα συνάρτηση kmeans με αυτήν:

void kmeans(Vector** vectors, int n, Cluster* clusters, int k, int dimension, int max_iterations, unsigned int seed) {
    // ΣΚΛΗΡΟ ΟΡΙΟ: Αν το max_iterations είναι πολύ μεγάλο (π.χ. 1000), το κόβουμε στο 20 για να τελειώνει γρήγορα.
    // Αν θέλετε περισσότερη ακρίβεια αργότερα, αυξήστε το στο 50.
    int LIMIT = (max_iterations > 20) ? 20 : max_iterations; 
    if (LIMIT < 5) LIMIT = 5; // Τουλάχιστον 5 επαναλήψεις

    printf("[DEBUG] K-Means started. Points: %d, Clusters: %d, Max Iterations: %d\n", n, k, LIMIT);

    // 1. Initialize Centroids
    initialize_centroids_kmeans(vectors, n, clusters, k, dimension, seed);
    
    // 2. Iterate
    for (int iter = 0; iter < LIMIT; iter++) {
        // Reset cluster member counts before assignment
        for (int c = 0; c < k; c++) {
            clusters[c].member_count = 0;
        }
        
        // Assignment Step
        int changes = assign_to_clusters(vectors, n, clusters, k);
        
        // Debug Print: Δείχνει ότι ο αλγόριθμος "ζει" και προχωράει 
        printf("[DEBUG] Iteration %d/%d: %d points changed clusters\n", iter+1, LIMIT, changes);

        // Check Convergence
        if (changes == 0) {
            printf("[DEBUG] K-means converged early at iteration %d\n", iter+1);
            break;
        }
        
        // Update Centroids Step
        update_centroids(vectors, clusters, k, dimension);
    }

    printf("[DEBUG] K-Means finished.\n");
}