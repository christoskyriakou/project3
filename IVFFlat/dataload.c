

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "kmeans.h"
#include "ivfflat.h"

// Dataset Loading Functions

// Byte swap for big-endian to little-endian conversion
uint32_t swap_endian_32(uint32_t val) {
    return ((val & 0x000000FF) << 24) |
           ((val & 0x0000FF00) << 8) |
           ((val & 0x00FF0000) >> 8) |
           ((val & 0xFF000000) >> 24);
}

// Read 32-bit integer (Big-Endian for MNIST)
uint32_t read_int32_be(FILE* f) {
    uint32_t val;
    if (fread(&val, sizeof(uint32_t), 1, f) != 1) {
        return 0;
    }
    return swap_endian_32(val);
}

// Read 32-bit integer (Little-Endian for SIFT)
uint32_t read_int32_le(FILE* f) {
    uint32_t val;
    if (fread(&val, sizeof(uint32_t), 1, f) != 1) {
        return 0;
    }
    return val;
}

// Load MNIST dataset
Vector** load_mnist(const char* filename, int* count, int* dimension, int isQuery) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    uint32_t magic = read_int32_be(f);
    if (magic != 2051) {
        fprintf(stderr, "Error: Invalid MNIST file format (magic=%u)\n", magic);
        fclose(f);
        return NULL;
    }
    
    uint32_t num_images = read_int32_be(f);
    uint32_t rows = read_int32_be(f);
    uint32_t cols = read_int32_be(f);
    
    *count = num_images;
    if (isQuery) {
        *count = 50;
    } else {
        *count = 500;
    }
    
    *dimension = rows * cols;
    
    printf("Loading MNIST: %u images of %ux%u (dim=%d)\n", 
           num_images, rows, cols, *dimension);
    
    Vector** vectors = (Vector**)malloc(*count * sizeof(Vector*));
    if (!vectors) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    for (int i = 0; i < *count; i++) {
        vectors[i] = create_vector(*dimension, i);
        if (!vectors[i]) {
            fprintf(stderr, "Error: Cannot create vector %d\n", i);
            fclose(f);
            return NULL;
        }
        
        for (int j = 0; j < *dimension; j++) {
            uint8_t pixel;
            if (fread(&pixel, sizeof(uint8_t), 1, f) != 1) {
                fprintf(stderr, "Error: Cannot read pixel data\n");
                fclose(f);
                return NULL;
            }
            vectors[i]->coords[j] = (float)pixel;
        }
        
        if ((i + 1) % *count == 0) {
            printf("  Loaded %d/%u images\n", i + 1, *count);
        }
    }
    
    fclose(f);
    printf("MNIST dataset loaded successfully\n\n");
    return vectors;
}

// Load SIFT dataset
Vector** load_sift(const char* filename, int* count, int* dimension, int isQuery) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    // Read first dimension value
    uint32_t dim;
    if (fread(&dim, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "Error: Cannot read dimension\n");
        fclose(f);
        return NULL;
    }
    *dimension = dim;
    
    // Count total vectors by file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    int bytes_per_vector = (dim + 1) * sizeof(float);
    *count = file_size / bytes_per_vector;
    
    // Less vectors than original file to save time
    if (isQuery) {
        *count = 1000;
    } else {
        *count = 10000;
    }
    
    
    
    printf("Loading SIFT: %d vectors of dimension %d\n", *count, *dimension);
    
    fseek(f, 0, SEEK_SET);
    
    Vector** vectors = (Vector**)malloc(*count * sizeof(Vector*));
    if (!vectors) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(f);
        return NULL;
    }
    
    for (int i = 0; i < *count; i++) {
        uint32_t d;
        if (fread(&d, sizeof(uint32_t), 1, f) != 1) {
            fprintf(stderr, "Error: Cannot read dimension for vector %d\n", i);
            break;
        }
        
        vectors[i] = create_vector(*dimension, i);
        if (!vectors[i]) {
            fprintf(stderr, "Error: Cannot create vector %d\n", i);
            break;
        }
        
        if (fread(vectors[i]->coords, sizeof(float), *dimension, f) != *dimension) {
            fprintf(stderr, "Error: Cannot read vector data %d\n", i);
            break;
        }
        
        if ((i + 1) % 100000 == 0) {
            printf("  Loaded %d/%d vectors\n", i + 1, *count);
        }
    }
    
    fclose(f);
    printf("SIFT dataset loaded successfully\n\n");
    return vectors;
}

// Evaluation Metrics

float calculate_average_approximation_factor(Neighbor* approx, Neighbor* exact, 
                                             int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        if (exact[i].distance > 0.0f) {
            sum += approx[i].distance / exact[i].distance;
        } else {
            sum += 1.0f;
        }
    }
    return count > 0 ? sum / count : 1.0f;
}

float calculate_recall_at_n(Neighbor* approx, Neighbor* exact, int N) {
    int matches = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (approx[i].id == exact[j].id) {
                matches++;
                break;
            }
        }
    }
    return N > 0 ? (float)matches / N : 0.0f;
}
