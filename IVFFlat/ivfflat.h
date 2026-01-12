IVFFlat* create_ivfflat(int num_clusters, int dimension);

void build_ivfflat(IVFFlat* index, Vector** vectors, int n, unsigned int seed);

int compare_neighbors(const void* a, const void* b);

Neighbor* search_ivfflat(IVFFlat* index, Vector** vectors, Vector* query, int nprobe, int N, int* result_count);

Neighbor* range_search_ivfflat(IVFFlat* index, Vector** vectors, Vector* query, float radius, int* result_count);

void free_ivfflat(IVFFlat* index);