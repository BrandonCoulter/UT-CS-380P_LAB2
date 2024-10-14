#include "kmeans_cuda_basic.h"

__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int cuda_kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts)
{
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Corrected member access: use '.' instead of '->'
    a = clusters[0].pointID;  // Use '.' because clusters[0] is a struct, not a pointer
    b = clusters[1].pointID;  // Use '.' because clusters[1] is a struct, not a pointer

    // Copy data from host to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // Call the CUDA kernel
    add<<<1,1>>>(d_a, d_b, d_c);

    // Copy result from device to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}