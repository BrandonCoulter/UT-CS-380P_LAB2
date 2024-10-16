#ifndef _KMEANS_CUDA_SHMEM_H
#define _KMEANS_CUDA_SHMEM_H

#include <cstdio>
#include <math.h>
#include <cfloat>
#include <iostream>
#include <cuda_runtime.h>

#include "point.h"
#include "centroid.h"
#include "argparse.h"
#include "kmeans_GPU_utils.h"

namespace cuda_shmem
{
    __global__ void assign_points_to_clusters(struct Centroid* clusters, struct Point* points, struct options_t* opts, int* converged);

    void cuda_kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts);
}

#endif