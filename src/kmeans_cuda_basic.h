#ifndef _KMEANS_CUDA_H
#define _KMEANS_CUDA_H

#include <cuda_runtime.h>

#include "point.h"
#include "centroid.h"
#include "argparse.h"

__global__ void add(int *a, int *b, int *c);

int cuda_kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts);

#endif