

KMeans* kmeans_create(int k, int dim);

void kmeans_init_centroids(KMeans *km, Vector *data, int n, unsigned int seed);

void kmeans_train(KMeans *km, Vector *data, int n, int max_iter);

void kmeans_free(KMeans *km);

float compute_silhouette_score(Vector *data, int n, int *assignments, Vector *centroids, int k, int dim);

float compute_silhouette_sample(Vector *data, int n, int *assignments, Vector *centroids, int k, int dim, int sample_size);

int find_optimal_k_silhouette(Vector *data, int n, int dim, int k_min, int k_max, int sample_size, unsigned int seed);