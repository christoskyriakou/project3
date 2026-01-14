#include "kmeans.h"
#include "ivfflat.h"
#include "dataload.h"


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Brute Force Search (Ground Truth)

Neighbor* brute_force_search(Vector** vectors, int n, Vector* query, int N, int* result_count) {
    Neighbor* all = (Neighbor*)malloc(n * sizeof(Neighbor));
    if (!all) return NULL;
    
    for (int i = 0; i < n; i++) {
        all[i].id = i;
        all[i].distance = euclidean_distance(query, vectors[i]);
    }
    
    qsort(all, n, sizeof(Neighbor), compare_neighbors);
    
    *result_count = (N < n) ? N : n;
    Neighbor* results = (Neighbor*)malloc(*result_count * sizeof(Neighbor));
    if (results) {
        memcpy(results, all, *result_count * sizeof(Neighbor));
    }
    
    free(all);
    return results;
}

Neighbor* brute_force_range_search(Vector** vectors, int n, Vector* query, float radius, int* result_count) {
    int max_results = 10000;
    Neighbor* results = (Neighbor*)malloc(max_results * sizeof(Neighbor));
    *result_count = 0;
    
    for (int i = 0; i < n; i++) {
        float dist = euclidean_distance(query, vectors[i]);
        
        if (dist <= radius) {
            if (*result_count >= max_results) {
                max_results *= 2;
                results = (Neighbor*)realloc(results, max_results * sizeof(Neighbor));
            }
            results[*result_count].id = i;
            results[*result_count].distance = dist;
            (*result_count)++;
        }
    }
    
    return results;
}

int main(int argc, char** argv) {
    char* input_file = NULL;
    char* query_file = NULL;
    char* output_file = "output.txt";
    char* type = "mnist";
    int kclusters = 50;
    int nprobe = 5;
    int N = 1;
    float R = 2000.0f;
    unsigned int seed = 1;
    int do_range_search = 1;
    
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
        else if (strcmp(argv[i], "-range") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "false") == 0) {
                do_range_search = 0;
            }
        }
        else if (strcmp(argv[i], "-ivfflat") == 0) {
            // Flag for IVFFlat mode
        }
    }
    
    if (!input_file || !query_file) {
        fprintf(stderr, "Usage: %s -d <input> -q <query> -o <output> -type <sift|mnist> "
                "-kclusters <int> -nprobe <int> -N <int> -R <double> -seed <int> "
                "-range <true|false> -ivfflat\n", argv[0]);
        return 1;
    }
    
    // Adjust R for SIFT if needed
    // if (strcmp(type, "sift") == 0 && R > 1000.0f) {
    //     R = 300.0f;
    // }
    
    printf("===============================================\n");
    printf("IVFFlat Vector Search\n");
    printf("===============================================\n");
    printf("Configuration:\n");
    printf("  Dataset type:    %s\n", type);
    printf("  Input file:      %s\n", input_file);
    printf("  Query file:      %s\n", query_file);
    printf("  Output file:     %s\n", output_file);
    printf("  K-clusters:      %d\n", kclusters);
    printf("  N-probe:         %d\n", nprobe);
    printf("  N neighbors:     %d\n", N);
    printf("  Radius R:        %.2f\n", R);
    printf("  Seed:            %u\n", seed);
    printf("  Range search:    %s\n", do_range_search ? "enabled" : "disabled");
    printf("===============================================\n\n");
    
    int data_count, query_count, dimension;
    Vector** data_vectors;
    Vector** query_vectors;
    
    if (strcmp(type, "mnist") == 0) {
        data_vectors = load_mnist(input_file, &data_count, &dimension, 0);
        query_vectors = load_mnist(query_file, &query_count, &dimension, 1);
    } else if (strcmp(type, "sift") == 0) {
        data_vectors = load_sift(input_file, &data_count, &dimension, 0);
        query_vectors = load_sift(query_file, &query_count, &dimension, 1);
    } else {
        fprintf(stderr, "Error: Unknown type '%s'. Use 'mnist' or 'sift'\n", type);
        return 1;
    }
    
    if (!data_vectors || !query_vectors) {
        fprintf(stderr, "Error: Failed to load datasets\n");
        return 1;
    }
    
    printf("Dataset loaded: %d vectors (dimension: %d)\n", data_count, dimension);
    printf("Query set loaded: %d vectors\n\n", query_count);
    
    printf("Building IVFFlat index...\n");
    clock_t start = clock();
    IVFFlat* index = create_ivfflat(kclusters, dimension);
    if (!index) {
        fprintf(stderr, "Error: Failed to create IVFFlat index\n");
        return 1;
    }
    build_ivfflat(index, data_vectors, data_count, seed);
    clock_t end = clock();
    double build_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Index built in %.2f seconds\n\n", build_time);
    
    FILE* out = fopen(output_file, "w");
    if (!out) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
        return 1;
    }
    
    fprintf(out, "IVFFlat\n");
    
    printf("Processing queries...\n");
    double total_approx_time = 0.0;
    double total_exact_time = 0.0;
    float total_af = 0.0f;
    float total_recall = 0.0f;
    int queries_processed = 0;
    
    for (int q = 0; q < query_count; q++) {
        fprintf(out, "Query: %d\n", q);
        
        // Approximate search
        start = clock();
        int approx_count;
        Neighbor* approx_results = search_ivfflat(index, data_vectors, query_vectors[q], nprobe, N, &approx_count);
        end = clock();
        double approx_time = (double)(end - start) / CLOCKS_PER_SEC;
        total_approx_time += approx_time;
        
        // Exact search (ground truth)
        start = clock();
        int exact_count;
        Neighbor* exact_results = brute_force_search(data_vectors, data_count, query_vectors[q], N, &exact_count);
        end = clock();
        double exact_time = (double)(end - start) / CLOCKS_PER_SEC;
        total_exact_time += exact_time;
        
        // Write N nearest neighbors
        int max_neighbors = (approx_count < N) ? approx_count : N;
        for (int i = 0; i < max_neighbors; i++) {
            fprintf(out, "Nearest neighbor-%d: %d\n", i + 1, approx_results[i].id);
            fprintf(out, "distanceApproximate: %.6f\n", approx_results[i].distance);
            fprintf(out, "distanceTrue: %.6f\n", 
                    (i < exact_count) ? exact_results[i].distance : 0.0f);
        }
        
        // Range search (using brute force for accurate results or approx?)
        if (do_range_search) {
            int range_count;
            //Neighbor* range_results = brute_force_range_search(data_vectors, data_count, query_vectors[q], R, &range_count); 
            Neighbor* range_results = range_search_ivfflat(index, data_vectors, query_vectors[q], R, &range_count); 
            
            fprintf(out, "R-near neighbors %d:\n", range_count);
            if (range_count > 0) {
                for (int i = 0; i < range_count; i++) {
                    fprintf(out, "%d\n", range_results[i].id);
                }
            }
            
            free(range_results);
        }
        
        if (approx_count > 0 && exact_count > 0) {
            int eval_count = (approx_count < N) ? approx_count : N;
            float af = calculate_average_approximation_factor(approx_results, exact_results, eval_count);
            float recall = calculate_recall_at_n(approx_results, exact_results, eval_count);
            total_af += af;
            total_recall += recall;
            queries_processed++;
        }
        
        free(approx_results);
        free(exact_results);
        
        if ((q + 1) % 100 == 0 || q == query_count - 1) {
            printf("  Processed %d/%d queries\n", q + 1, query_count);
        }
    }
    
    double avg_approx_time = total_approx_time / query_count;
    double avg_exact_time = total_exact_time / query_count;
    double qps = query_count / total_approx_time;
    float avg_af = queries_processed > 0 ? total_af / queries_processed : 1.0f;
    float avg_recall = queries_processed > 0 ? total_recall / queries_processed : 0.0f;
    
    fprintf(out, "Average AF: %.6f\n", avg_af);
    fprintf(out, "Recall@N: %.6f\n", avg_recall);
    fprintf(out, "QPS: %.2f\n", qps);
    fprintf(out, "tApproximateAverage: %.6f\n", avg_approx_time);
    fprintf(out, "tTrueAverage: %.6f\n", avg_exact_time);
    
    fclose(out);
    
    printf("\n===============================================\n");
    printf("Results Summary\n");
    printf("===============================================\n");
    printf("Queries processed:   %d\n", query_count);
    printf("Average AF:          %.6f\n", avg_af);
    printf("Recall@%d:           %.6f (%.1f%%)\n", N, avg_recall, avg_recall * 100);
    printf("QPS:                 %.2f queries/sec\n", qps);
    printf("Avg approx time:     %.6f sec\n", avg_approx_time);
    printf("Avg exact time:      %.6f sec\n", avg_exact_time);
    printf("Speedup:             %.2fx\n", avg_exact_time / avg_approx_time);
    printf("===============================================\n");
    printf("Results written to: %s\n", output_file);
    
    free_ivfflat(index);
    for (int i = 0; i < data_count; i++) {
        free_vector(data_vectors[i]);
    }
    for (int i = 0; i < query_count; i++) {
        free_vector(query_vectors[i]);
    }
    free(data_vectors);
    free(query_vectors);
    
    return 0;
}