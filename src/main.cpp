// #include <iostream>
#include <iomanip>
#include <io.h>
#include <chrono>

#include "kmeans.h"
#include "point.h"
#include "centroid.h"
#include "argparse.h"
#include "randcentroid.h"
#include "kmeans_cuda_basic.h"

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // std::cout << std::setprecision(18); // Set the persion of output stream

#ifdef __PRINT__
    std::cout << "N_CLUSTERS (k): " << opts.n_clusters << std::endl;
    std::cout << "N_DIMS (d): " << opts.n_dims << std::endl;
    std::cout << "IN_FILE (i): " << opts.in_file << std::endl;
    std::cout << "MAX_ITER (m): " << opts.max_iter << std::endl;
    std::cout << "THRESHOLD (t): " << opts.threshold << std::endl;
    std::cout << "SEED (s): " << opts.seed << std::endl;
    std::cout << "PRINT_CENT (c): " << opts.print_cent << std::endl;
    std::cout << "RUN CUDA (x): " << opts.run_cuda << std::endl;
    std::cout << "RUN SHMEM (y): " << opts.run_shmem << std::endl;
    std::cout << "RUN THRUST (z): " << opts.run_thrust << std::endl;
#endif

    // Populate points
    opts.n_points = get_n_points(&opts);
    struct Point* points = (struct Point*) malloc(opts.n_points * sizeof(struct Point));
    read_file(&opts, points); // Read file and populate the points
    // Generate "random" clusters (Sudo random because intial cluster locations can change the output significantly)
    kmeans_srand(opts.seed); // Generate clusters based on cmd passed seed
    struct Centroid* clusters = gen_initial_centroid(&opts, points);

#ifdef __PRINT__
    std::cout << "Starting centroid points" << std::endl;
    for(int k = 0; k < opts.n_clusters; k++)
    {
        std::cout << "Cluster #" << k << " start index: " << clusters[k].pointID << std::endl;
    }
#endif

#if defined(__PRINT__) && defined(__VERBOSE__)
    std::cout << "POINTS: " << std::endl;
    for(int p = 0; p < opts.n_points; p++)
    {
        points[p].print(&opts);
    }
    std::cout << "CLUSTERS: " << std::endl;
    for(int k = 0; k < opts.n_clusters; k++)
    {
        clusters[k].print(&opts);
    }
#endif
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    kmeans(clusters, points, &opts);

    //End timer and print out elapsed
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if(opts.run_cuda)
    {
        std::cout << "time: " << diff.count() << " | " << clusters[0].pointID << " + "<< clusters[1].pointID << " = " << cuda_kmeans(clusters, points, &opts) << std::endl;
    }
    else
    {
        std::cout << "time: " << diff.count() << std::endl;
    }

    if(opts.print_cent)
    {   
        for(int k = 0; k < opts.n_clusters; k++)
        {
            // clusters[k].print(&opts);
            std::cout << clusters[k].clusterID << " ";
            for(int d = 0; d < opts.n_dims; d++)
            {
                std::cout << clusters[k].position[d] << " ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cout << "clusters:";

        for(int p = 0; p < opts.n_points; p++)
        {
            std::cout << " " << points[p].clusterID;
        }
        std::cout << std::endl;
    }
  
    free(points);
    free(clusters);
}