#ifndef _KMEANS_CUDA_H
#define _KMEANS_CUDA_H

#include <math.h>
#include <cfloat>
#include <iostream>
#include <cuda_runtime.h>

#include "point.h"
#include "centroid.h"
#include "argparse.h"
#include "kmeans_GPU_utils.h"

namespace cuda_basic
{
    __global__ void assign_points_to_clusters(struct Centroid* clusters, struct Point* points, struct options_t* opts, int* converged);
    __global__ void add_new_centroid_position(struct Centroid* clusters, struct Point* points, struct options_t* opts);
    __global__ void find_new_centroid_position(struct Centroid* clusters, struct Point* points, struct options_t* opts);
    __global__ void reset_centroids(struct Centroid* clusters, struct options_t* opts);
    
    void cuda_kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts);
}
//__device__ void squared_distance(double* c_pos, double* p_pos, double* dis, int n_dims);

#endif