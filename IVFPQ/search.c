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
#include "ivfpq.h"

// Brute Force Search (Ground Truth)
SearchResult* brute_force_search(Vector *dataset, int dataset_size, Vector *query, int N, int *result_count) {
    SearchResult *results = (SearchResult*)malloc(dataset_size * sizeof(SearchResult));
    
    for (int i = 0; i < dataset_size; i++) {
        results[i].id = dataset[i].id;
        results[i].distance = euclidean_distance(query->coords, dataset[i].coords, query->dim);
    }
    
    qsort(results, dataset_size, sizeof(SearchResult), compare_results);
    
    int final_count = (N < dataset_size) ? N : dataset_size;
    SearchResult *final_results = (SearchResult*)malloc(final_count * sizeof(SearchResult));
    memcpy(final_results, results, final_count * sizeof(SearchResult));
    
    free(results);
    *result_count = final_count;
    return final_results;
}

int main(int argc, char *argv[]) {
    char* input_file = NULL;
    char* query_file = NULL;
    char* output_file = "output.txt";
    char* type = "mnist";
    int kclusters = 50;
    int nprobe = 5;
    int M = 16;
    int nbits = 8;
    int N = 1;
    float R = 2000.0f;
    unsigned int seed = 1;
    int range_search = 1;
    int optimize_k = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        }
        else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            query_file = argv[++i];
        }
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        }
        else if (strcmp(argv[i], "-type") == 0 && i + 1 < argc) {
            type = argv[++i];
        }
        else if (strcmp(argv[i], "-kclusters") == 0 && i + 1 < argc) {
            kclusters = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-nprobe") == 0 && i + 1 < argc) {
            nprobe = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-R") == 0 && i + 1 < argc) {
            R = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-nbits") == 0 && i + 1 < argc) {
            nbits = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-optimize_k") == 0 && i + 1 < argc) {
            optimize_k = 1;
            printf("found it\n");
        }
        else if (strcmp(argv[i], "-range") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "false") == 0) {
                range_search = 0;
            }
        }
        else if (strcmp(argv[i], "-ivfpq") == 0) {
            // Flag for IVFFlat mode
        }
    }
    
    if (!input_file || !query_file) {
        fprintf(stderr, "Usage: %s -d <input> -q <query> -o <output> -type <sift|mnist> "
                "-kclusters <int> -nprobe <int> -N <int> -R <double> -seed <int> "
                "-range <true|false> -ivfflat\n", argv[0]);
        return 1;
    }
    
    Vector *dataset = NULL;
    int dataset_size = 0, dataset_dim = 0;
    
    if (strcmp(type, "mnist") == 0) {
        dataset = load_mnist(input_file, &dataset_size, &dataset_dim, 0);
    } else if (strcmp(type, "sift") == 0) {
        dataset = load_sift(input_file, &dataset_size, &dataset_dim, 0);
    } else {
        printf("Unknown type: %s\n", type);
        return 1;
    }
    
    if (!dataset) {
        printf("Failed to load dataset\n");
        return 1;
    }
    
    Vector *queries = NULL;
    int query_count = 0, query_dim = 0;
    
    if (strcmp(type, "mnist") == 0) {
        queries = load_mnist(query_file, &query_count, &query_dim, 1);
    } else {
        queries = load_sift(query_file, &query_count, &query_dim, 1);
    }
    
    if (!queries) { 
        printf("Failed to load queries\n");    
        free_vectors(dataset, dataset_size);
        return 1;
    }
    
    printf("\n");
    
    double start_time = get_time();

    int k_min = 20;
    int k_max = 500;
    
    // Optimize k if requested
    int optimal_kclusters = kclusters;
    if (optimize_k) {
        // Use a sample for silhouette computation (max 5000 points)
        int sample_size = (dataset_size < 5000) ? dataset_size : 5000;
        optimal_kclusters = find_optimal_k_silhouette(dataset, dataset_size, dataset_dim, k_min, k_max, sample_size, seed);
        kclusters = optimal_kclusters;
        printf("Using optimized kclusters = %d\n\n", optimal_kclusters);
    }
    
    IVFPQ *index = ivfpq_create(kclusters, M, nbits, dataset_dim);
    
    printf("\nTraining IVFPQ index...\n");
    ivfpq_train(index, dataset, dataset_size, seed);
    
    printf("\nAdding vectors to index...\n");
    ivfpq_add(index, dataset, dataset_size);
    
    double train_time = get_time() - start_time;
    printf("\nIndex built in %.2f seconds\n", train_time);
    
    // Compute and display Silhouette Score for the built index
    // printf("\n=== Computing Silhouette Score for Built Index ===\n"); 
    
    // Get assignments for all vectors in the dataset
    int *final_assignments = (int*)malloc(dataset_size * sizeof(int));
    for (int i = 0; i < dataset_size; i++) {
        float min_dist = FLT_MAX;
        int best_cluster = 0;
        
        for (int k = 0; k < index->kclusters; k++) {
            float dist = euclidean_distance(dataset[i].coords, index->coarse_quantizer->centroids[k].coords, dataset_dim);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        final_assignments[i] = best_cluster;
    }
    
    int silhouette_sample_size = (dataset_size < 5000) ? dataset_size : 5000;

    double silhouette_start = get_time();
    float silhouette_score = compute_silhouette_sample(dataset, dataset_size, final_assignments, index->coarse_quantizer->centroids, index->kclusters, dataset_dim, silhouette_sample_size);
    double silhouette_time = get_time() - silhouette_start;
    
    printf("\n");
    printf("Silhouette Score: %.6f\n", silhouette_score);
    printf("Computation time: %.2f seconds\n", silhouette_time);
    // printf("\nInterpretation:\n");
    // if (silhouette_score > 0.7) {
    //     printf("  Excellent clustering (>0.7)\n");
    // } else if (silhouette_score > 0.5) {
    //     printf("  Good clustering (0.5-0.7)\n");
    // } else if (silhouette_score > 0.25) {
    //     printf("  Weak clustering (0.25-0.5)\n");
    // } else if (silhouette_score > 0) {
    //     printf("  Poor clustering (0-0.25)\n");
    // } else {
    //     printf("  Very poor clustering (<0)\n");
    // }
    
    // Compute cluster statistics
    //printf("\nCluster Statistics:\n");
    int *cluster_sizes = (int*)calloc(index->kclusters, sizeof(int));
    int min_size = dataset_size, max_size = 0;
    
    for (int i = 0; i < dataset_size; i++) {
        cluster_sizes[final_assignments[i]]++;
    }
    
    for (int k = 0; k < index->kclusters; k++) {
        if (cluster_sizes[k] < min_size) min_size = cluster_sizes[k];
        if (cluster_sizes[k] > max_size) max_size = cluster_sizes[k];
    }
    
    float avg_size = (float)dataset_size / index->kclusters;
    
    // Compute standard deviation
    float variance = 0.0;
    for (int k = 0; k < index->kclusters; k++) {
        float diff = cluster_sizes[k] - avg_size;
        variance += diff * diff;
    }
    variance /= index->kclusters;
    //float std_dev = sqrtf(variance);
    
    //printf("  Average cluster size: %.1f\n", avg_size);
    //printf("  Min cluster size: %d\n", min_size);
    //printf("  Max cluster size: %d\n", max_size);
    //printf("  Std deviation: %.1f\n", std_dev);
    //printf("  Balance ratio: %.3f (max/avg)\n", max_size / avg_size);
    
    
    free(cluster_sizes);
    //printf("========================================\n");
    
    free(final_assignments);
    
    // Store silhouette score for later use
    //float index_silhouette_score = silhouette_score;
    
    // Open output file
    FILE *output = fopen(output_file, "w");
    if (!output) {
        printf("Error opening output file: %s\n", output_file);
        ivfpq_free(index);
        free_vectors(dataset, dataset_size);
        free_vectors(queries, query_count);
        return 1;
    }
    
    fprintf(output, "IVFPQ\n");
    // fprintf(output, "Index Configuration:\n");
    // fprintf(output, "  kclusters: %d\n", kclusters);
    // fprintf(output, "  nprobe: %d\n", nprobe);
    // fprintf(output, "  M: %d\n", M);
    // fprintf(output, "  nbits: %d\n", nbits);
    // fprintf(output, "  Silhouette Score: %.6f\n", index_silhouette_score);
    // fprintf(output, "\n");
    
    printf("\nProcessing queries...\n");
    double total_approx_time = 0.0;
    double total_exact_time = 0.0;
    double total_af = 0.0;
    double total_recall = 0.0;
    int processed_queries = 0;
    
    for (int q = 0; q < query_count; q++) {
        if ((q + 1) % 100 == 0 || q == 0) {
            printf("  Query: %d/%d\n", q + 1, query_count);
        }
        
        fprintf(output, "Query: %d\n", queries[q].id);
        
        double approx_start = get_time();
        int approx_count = 0;
        SearchResult *approx_results = ivfpq_search(index, &queries[q], nprobe, N, &approx_count);
        double approx_time = get_time() - approx_start;
        total_approx_time += approx_time;
        
        double exact_start = get_time();
        int exact_count = 0;
        SearchResult *exact_results = brute_force_search(dataset, dataset_size, &queries[q], N, &exact_count);
        double exact_time = get_time() - exact_start;
        total_exact_time += exact_time;
        
        for (int i = 0; i < approx_count && i < N; i++) {
            fprintf(output, "Nearest neighbor-%d: %d\n", i + 1, approx_results[i].id);
            fprintf(output, "distanceApproximate: %.6f\n", approx_results[i].distance);
            
            if (i < exact_count) {
                fprintf(output, "distanceTrue: %.6f\n", exact_results[i].distance);
            } else {
                fprintf(output, "distanceTrue: N/A\n");
            }
        }
        
        for (int i = approx_count; i < N; i++) {
            fprintf(output, "Nearest neighbor-%d: N/A\n", i + 1);
            fprintf(output, "distanceApproximate: N/A\n");
            if (i < exact_count) {
                fprintf(output, "distanceTrue: %.6f\n", exact_results[i].distance);
            } else {
                fprintf(output, "distanceTrue: N/A\n");
            }
        }
        
        if (range_search) {
            int range_count = 0;
            SearchResult *range_results = ivfpq_range_search(index, &queries[q], R, &range_count);
            
            fprintf(output, "R-near neighbors:\n");
            if (range_count > 0) {
                for (int i = 0; i < range_count; i++) {
                    fprintf(output, "%d\n", range_results[i].id);
                }
            }
            
            free(range_results);
        }
        
        float af = 0.0;
        if (approx_count > 0 && exact_count > 0) {
            af = approx_results[0].distance / exact_results[0].distance;
        } else if (exact_count > 0) {
            af = 100.0; // Large penalty if no result found
        }
        
        float recall = 0.0;
        if (approx_count > 0 && exact_count > 0) {
            recall = compute_recall(approx_results, approx_count, exact_results, exact_count);
        }
        
        total_af += af;
        total_recall += recall;
        processed_queries++;
        
        free(approx_results);
        free(exact_results);
    }
    
    double avg_af = processed_queries > 0 ? total_af / processed_queries : 0.0;
    double avg_recall = processed_queries > 0 ? total_recall / processed_queries : 0.0;
    double avg_approx_time = processed_queries > 0 ? total_approx_time / processed_queries : 0.0;
    double avg_exact_time = processed_queries > 0 ? total_exact_time / processed_queries : 0.0;
    double qps = processed_queries > 0 ? processed_queries / total_approx_time : 0.0;
    
    fprintf(output, "Average AF: %.6f\n", avg_af);
    fprintf(output, "Recall@N: %.6f\n", avg_recall);
    fprintf(output, "QPS: %.2f\n", qps);
    fprintf(output, "tApproximateAverage: %.6f\n", avg_approx_time);
    fprintf(output, "tTrueAverage: %.6f\n", avg_exact_time);
    
    fclose(output);
    
    printf("\n=== Results Summary ===\n");
    printf("Average Approximation Factor: %.6f\n", avg_af);
    printf("Recall@%d: %.6f\n", N, avg_recall);
    printf("Queries Per Second: %.2f\n", qps);
    printf("Average Approximate Search Time: %.6f seconds\n", avg_approx_time);
    printf("Average Exact Search Time: %.6f seconds\n", avg_exact_time);
    printf("Speedup: %.2fx\n", avg_exact_time / avg_approx_time);
    printf("\nResults written to: %s\n", output_file);
    
    ivfpq_free(index);
    free_vectors(dataset, dataset_size);
    free_vectors(queries, query_count);
    
    return 0;
}