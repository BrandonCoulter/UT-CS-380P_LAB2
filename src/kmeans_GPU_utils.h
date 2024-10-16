#ifndef _KMEANS_GPU_UTILS_H
#define _KMEANS_GPU_UTILS_H

#define CUDA_CHECK_ERROR() {                                          \
    cudaError_t err = cudaGetLastError();                             \
    if (err != cudaSuccess) {                                         \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err),  \
                __FILE__, __LINE__);                                  \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

inline __device__ void squared_distance(double* c_pos, double* p_pos, double* dis, int n_dims)
{
    for(int d = 0; d < n_dims; d++)
    {
        *dis += (c_pos[d] - p_pos[d]) * (c_pos[d] - p_pos[d]);
    }
}

#endif