

typedef struct {
    float *coords;
    int dim;
    int id;
} Vector;

typedef struct {
    int *assignments;
    int size;
} Cluster;

typedef struct {
    Vector *centroids;
    Cluster *clusters;
    int k;
    int dim;
} KMeans;

typedef struct {
    unsigned char *codes;
    int *vector_ids;
    int count;
} InvertedList;

typedef struct {
    KMeans *coarse_quantizer;
    Vector **pq_centroids;
    InvertedList *inverted_lists;
    int kclusters;
    int M;
    int nbits;
    int subvec_dim;
    int dim;
} IVFPQ;

typedef struct {
    int id;
    float distance;
} SearchResult;

typedef struct {
    char *input_file;
    char *query_file;
    char *output_file;
    char *type;
    int kclusters;
    int nprobe;
    int M;
    int nbits;
    int N;
    float R;
    int seed;
    int range_search;
    int optimize_k;  // Flag to enable k optimization
    int k_min;       // Minimum k for optimization
    int k_max;       // Maximum k for optimization
} Arguments;

double get_time();

float euclidean_distance(float *a, float *b, int dim);

float asymmetric_distance(float *query_subvec, Vector *centroids, unsigned char code, int subvec_dim);

void init_random(unsigned int seed);

int rand_int(int max);

int compare_results(const void *a, const void *b);

Vector* load_mnist(const char *filename, int *count, int *dim, int isQuery);

Vector* load_sift(const char *filename, int *count, int *dim, int isQuery);

void free_vectors(Vector *vectors, int count);

float compute_recall(SearchResult *approx, int approx_count, SearchResult *exact, int exact_count);