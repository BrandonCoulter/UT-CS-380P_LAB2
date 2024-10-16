#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#include "io.h"
#include "kmeans.h"
#include "point.h"
#include "centroid.h"
#include "argparse.h"
#include "randcentroid.h"
#include "kmeans_cuda_basic.h"
#include "kmeans_cuda_shmem.h"

int main(int argc, char **argv)
{

    //Start e2e timer
    auto e2e_start = std::chrono::high_resolution_clock::now();

    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

#ifdef __PRINT__
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    printf("DEVICE PROPERTIES: \n");
    printf("Device name: %s\n", properties.name);
    printf("Total global memory: %lu\n", properties.totalGlobalMem);
    printf("Shared memory per block: %lu\n", properties.sharedMemPerBlock);
    printf("Registers per block: %d\n", properties.regsPerBlock);
    printf("Warp size: %d\n",  properties.warpSize);
    printf("Max threads per block: %d\n", properties.maxThreadsPerBlock);
    printf("Max threads dim: (%d, %d, %d)\n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
    printf("Max grid size: (%d, %d, %d)\n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);
    printf("Clock rate: %d\n", properties.clockRate);
    printf("Total constant memory: %lu\n", properties.totalConstMem);
    printf("Compute capability: %d.%d\n", properties.major, properties.minor);
    printf("Multiprocessor count: %d\n", properties.multiProcessorCount);
    printf("Kernel execution timeout: %s\n", (properties.kernelExecTimeoutEnabled ? "Enabled" : "Disabled"));

    printf("\nCMD Arguments");
    printf("N_CLUSTERS (k): %d\n", opts.n_clusters);
    printf("N_DIMS (d): %d\n", opts.n_dims);
    printf("IN_FILE (i): %s\n", opts.in_file);
    printf("MAX_ITER (m): %d\n", opts.max_iter);
    printf("THRESHOLD (t): %e\n", opts.threshold);
    printf("SEED (s): %d\n", opts.seed);
    printf("RUN_CUDA (x): %d\n", opts.run_cuda);
    printf("RUN_SHMEM (y): %d\n", opts.run_shmem);
    printf("RUN_THRUST (z): %d\n", opts.run_thrust);
#endif

    // Populate points
    opts.n_points = get_n_points(&opts);
    struct Point* points = (struct Point*) malloc(opts.n_points * sizeof(struct Point));
    read_file(&opts, points); // Read file and populate the points
    // Generate "random" clusters (Sudo random because intial cluster locations can change the output significantly)
    kmeans_srand(opts.seed); // Generate clusters based on cmd passed seed
    struct Centroid* clusters = gen_initial_centroid(&opts, points);

#ifdef __PRINT__
    printf("Starting Centroid Points\n");
    for(int k = 0; k < opts.n_clusters; k++)
    {
        printf("Cluster #%d start index: %d\n", k, clusters[k].pointID);
    }
#endif

#if defined(__PRINT__) && defined(__VERBOSE__)
    printf("POINTS: \n");
    for(int p = 0; p < opts.n_points; p++)
    {
        points[p].print(&opts);
    }
    printf("CLSUTERS: \n");
    for(int k = 0; k < opts.n_clusters; k++)
    {
        clusters[k].print(&opts);
    }
#endif

    if(opts.run_cuda)
    {
#ifndef __PRINT__
        printf("Cuda Basic | ");
#endif

        cuda_basic::cuda_kmeans(clusters, points, &opts);

#ifdef __PRINT__
        printf("Cuda Basic | ");
#endif
    }
    else if(opts.run_shmem)
    {
#ifndef __PRINT__
        printf("Cuda SHMEM | ");
#endif

        cuda_shmem::cuda_kmeans(clusters, points, &opts);

#ifdef __PRINT__
        printf("Cuda SHMEM | ");
#endif
    }
    else if(opts.run_thrust)
    {
        //TODO: Add Thrust implementation
    }
    else
    {
#ifndef __PRINT__
        printf("Sequential | ");
#endif
        kmeans(clusters, points, &opts);
#ifdef __PRINT__
        printf("Sequential | ");
#endif

    }

    //End e2e timer
    auto e2e_end = std::chrono::high_resolution_clock::now();
    auto total_e2e_time = std::chrono::duration_cast<std::chrono::milliseconds>(e2e_end - e2e_start);

    double e2e_elapsed_time = total_e2e_time.count();

    printf("E2E: %f\n", e2e_elapsed_time);

    if(opts.print_cent)
    {   
        for(int k = 0; k < opts.n_clusters; k++)
        {
            printf("%d ",  clusters[k].clusterID);
            for(int d = 0; d < opts.n_dims; d++)
            {
                printf("%lf ", clusters[k].position[d]);
            }
            printf("\n");
        }
    }
    else
    {
        printf("clusters:");

        for(int p = 0; p < opts.n_points; p++)
        {
            printf(" %d", points[p].clusterID);
        }
        printf("\n");
    }
  
    free(points);
    free(clusters);
}