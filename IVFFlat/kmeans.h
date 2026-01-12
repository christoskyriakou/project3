

typedef struct {
    float* coords;      // Vector coordinates
    int dimension;      // Vector dimension
    int id;            // Vector ID
} Vector;

typedef struct {
    Vector* centroid;   // Cluster centroid
    int* members;       // Array of vector IDs in this cluster
    int member_count;   // Number of members
    int capacity;       // Allocated capacity
} Cluster;

typedef struct {
    Cluster* clusters;  // Array of clusters
    int num_clusters;   // Number of clusters (k)
    int dimension;      // Vector dimension
} IVFFlat;

typedef struct {
    int id;            // Vector ID
    float distance;    // Distance to query
} Neighbor;

float euclidean_distance(const Vector* v1, const Vector* v2);

Vector* create_vector(int dimension, int id);

void copy_vector(Vector* dest, const Vector* src);

void free_vector(Vector* v);

void initialize_centroids_kmeans(Vector** vectors, int n, Cluster* clusters, int k, int dimension, unsigned int seed);

int assign_to_clusters(Vector** vectors, int n, Cluster* clusters, int k);

void update_centroids(Vector** vectors, Cluster* clusters, int k, int dimension);

float calculate_silhouette(Vector** vectors, int n, Cluster* clusters, int k);

void kmeans(Vector** vectors, int n, Cluster* clusters, int k, int dimension, int max_iterations, unsigned int seed);