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
        cluster.position = points[index].position;
        cluster.pointID = -1;
        cluster.clusterID = i;
        cluster.is_cluster = true;
        // you should use the proper implementation of the following
        // code according to your data structure
        // centers[i] = points[index];
        clusters[i] = cluster;
    }

    return clusters;
}