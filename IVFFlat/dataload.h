#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint32_t swap_endian_32(uint32_t val);

uint32_t read_int32_be(FILE* f);

uint32_t read_int32_le(FILE* f);

Vector** load_mnist(const char* filename, int* count, int* dimension, int isQuery);

Vector** load_sift(const char* filename, int* count, int* dimension, int isQuery);

float calculate_average_approximation_factor(Neighbor* approx, Neighbor* exact, int count);

float calculate_recall_at_n(Neighbor* approx, Neighbor* exact, int N);