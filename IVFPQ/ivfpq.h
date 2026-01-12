
IVFPQ* ivfpq_create(int kclusters, int M, int nbits, int dim);

void ivfpq_train(IVFPQ *index, Vector *data, int n, unsigned int seed);
void ivfpq_add(IVFPQ *index, Vector *data, int n);
SearchResult* ivfpq_search(IVFPQ *index, Vector *query, int nprobe, int N, int *result_count);
SearchResult* ivfpq_range_search(IVFPQ *index, Vector *query, float radius, int *result_count);
void ivfpq_free(IVFPQ *index);
