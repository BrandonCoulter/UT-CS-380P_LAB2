#include <randcentroid.h>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax + 1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

struct Centroid* gen_initial_centroid(struct options_t *opts, struct Point* points)
{
    struct Centroid* clusters = (struct Centroid*) malloc(opts->n_clusters*sizeof(struct Centroid));
    for (int i = 0; i < opts->n_clusters; i++){
        int index = kmeans_rand() % opts->n_points;
        struct Centroid cluster(*opts);
        // memcpy(cluster.position, points[index].position, opts->n_dims * sizeof(double));
        std::copy(points[index].position, points[index].position + opts->n_dims, cluster.position);
        cluster.pointID = index;
        cluster.clusterID = i;
        clusters[i] = cluster;
    }

    return clusters;
}