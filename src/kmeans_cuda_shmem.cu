#include "kmeans_cuda_shmem.h"

/*********************\
 DEVICE ONLY FUNCTIONS
\*********************/

__device__ void add_new_centroid_position(double* cluster_position, double* point_position, int n_dims)
{
    for(int d = 0; d < n_dims; d++)
    {
        atomicAdd(&cluster_position[d], point_position[d]);
    }
};

__device__ void find_new_centroid_position(double* cluster_position, int point_count, int n_dims)
{
    for(int d = 0; d < n_dims; d++)
    {
        cluster_position[d] = cluster_position[d] / point_count;

    }
};



__global__ void cuda_shmem::assign_points_to_clusters(struct Centroid* clusters, struct Point* points, struct options_t* opts, int* converged)
{
    extern __shared__ struct Centroid shared_clusters[];

    int gindex = threadIdx.x + blockIdx.x * blockDim.x; // index of the point to be processed
    int lindex = threadIdx.x; // Index of the cluster in shared memory

    if(lindex < opts->n_clusters)
    {
        shared_clusters[lindex] = clusters[lindex];
    }

    __syncthreads(); // Sync all threads in the block after loading shared memory

    if(gindex < opts->n_points)
    {
        // Declare local variables for cluster assignment
        int new_cluster_id = -999;
        double min_s_distance = 0.0;
        double min_e_distance = DBL_MAX;
        double s_distance; // Local sum of squared difference from the point to the selected cluster
        double e_distance; // Euclidean distance from the point to the selected cluster

        // Iterate clusters in shared memory to find the closest one
        for(int k = 0; k < opts->n_clusters; k++)
        {
            // Get the squared distance from point to shared memory cluster
            s_distance = 0.0;
            squared_distance(shared_clusters[k].position, points[gindex].position, &s_distance, opts->n_dims);

            // Get euclidean distance and check against min distance
            e_distance = sqrt(s_distance);
            if(e_distance < min_e_distance)
            {
                min_e_distance = e_distance;
                min_s_distance = s_distance;
                new_cluster_id = k;
            }

        }

        // Write the new positions to global memory
        add_new_centroid_position(clusters[new_cluster_id].new_position, points[gindex].position, opts->n_dims);

        // Add the sum of squared differences from point to cluster back into the global cluster sum of squared differences
        atomicAdd(&clusters[new_cluster_id].local_sum_squared_diff, min_s_distance);

        // Add 1 to the point count of the assigned global cluster
        atomicAdd(&clusters[new_cluster_id].point_count, 1);


        // Check point cluster assignment change and set convergence false
        if(points[gindex].clusterID != new_cluster_id)
        {
            points[gindex].clusterID = new_cluster_id; // Set the new cluster assignment
            atomicCAS(converged, 1, 0);
        }

    }

}



__global__ void cuda_shmem::calculate_centroid(struct Centroid* clusters, struct Point* points, struct options_t* opts, double* SSD)
{
    extern __shared__ struct Centroid shared_clusters[];

    int gindex = threadIdx.x + blockIdx.x * blockDim.x; // index of the point to be processed
    int lindex = threadIdx.x; // Index of the cluster in shared memory

    if(lindex < opts->n_clusters)
    {
        shared_clusters[lindex] = clusters[lindex];
    }

    __syncthreads(); // Sync all threads in the block after loading shared memory

    if(gindex < opts->n_clusters)
    {
        find_new_centroid_position(shared_clusters[gindex].new_position, shared_clusters[gindex].point_count, opts->n_dims);

        for(int d = 0; d < opts->n_dims; d++)
        {
            clusters[gindex].position[d] = shared_clusters[gindex].new_position[d];
            clusters[gindex].new_position[d] = 0.0;
        }

        atomicAdd(SSD, shared_clusters[gindex].local_sum_squared_diff);

        clusters[gindex].local_sum_squared_diff = 0.0; // Reset the local SSD
        clusters[gindex].point_count = 0; // Reset the point count
    }

}



void cuda_shmem::cuda_kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts)
{

    // Create a Cuda event based timer object for total elapsed time and iteration time
    CudaTimer timer;
    CudaTimer iter_timer;
    //float total_elapsed_time = 0;
    float average_iteration_time = 0;

    // Start Total Timer
    timer.start();

    struct Centroid* d_clusters; // Device copy of clusters
    struct Point* d_points; // Device copy of points
    
    // Allocate Device memory for clusters and points
    cudaMalloc((void**)&d_clusters, opts->n_clusters * sizeof(struct Centroid));
    cudaMalloc((void**)&d_points, opts->n_points * sizeof(struct Point));
    
    // Copy host to device memory for clusters and points
    cudaMemcpy(d_clusters, clusters, opts->n_clusters * sizeof(struct Centroid), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, points, opts->n_points * sizeof(struct Point), cudaMemcpyHostToDevice);
    
    // Select grid and block size
    int threads_per_block = 128;
    int blocks_per_grid = (opts->n_points + threads_per_block - 1) / threads_per_block;

    // Shared memory size
    int cluster_shmem_size = opts->n_clusters * sizeof(struct Centroid);
    
    // Initialize kmeans control variables
    int converged = 0; // int to control converged 0 = false, 1 = true
    int iter = opts->max_iter;
    double previous_sum_squared_difference= 0.0;
    double sum_squared_difference = 0.0;
    
    while(!converged && iter--) {

        iter_timer.start();
        
        // Set the convergence variable to true, if all conditions met, the loop will closes
        converged = 1;

        // Update previous SSD and reset SSD
        previous_sum_squared_difference = sum_squared_difference;
        sum_squared_difference = 0.0;
        
        // Assign points to clusters based on euclidean distance
        assign_points_to_clusters<<<blocks_per_grid, threads_per_block, cluster_shmem_size>>>(d_clusters, d_points, opts, &converged);
        CUDA_CHECK_ERROR();

        cudaDeviceSynchronize(); // Make sure that all kernals are completed
        
        // Calculate the new centroid position and reset for next iteration
        calculate_centroid<<<1, opts->n_clusters, cluster_shmem_size>>>(d_clusters, d_points, opts, &sum_squared_difference);
        CUDA_CHECK_ERROR();

        cudaDeviceSynchronize(); // Make sure that all kernals are completed

        // Copy device to host memory for clusters and points
        cudaMemcpy(clusters, d_clusters, opts->n_clusters * sizeof(struct Centroid), cudaMemcpyDeviceToHost);
        cudaMemcpy(points, d_points, opts->n_points * sizeof(struct Point), cudaMemcpyDeviceToHost);

        if(abs(sum_squared_difference - previous_sum_squared_difference) > opts->threshold)
        {
            converged = 0; // set convergence to false
        }

        iter_timer.stop();
        average_iteration_time += iter_timer.get_elapsed_time();

#ifdef __PRINT__

        // DEBUG PRINTING

        if(converged)
        {
            printf("P_SSD: %f | SSD: %f\n",previous_sum_squared_difference, sum_squared_difference);
        }

        printf("\nPost-assigned Points:\n");
        for(int p = 0; p < opts->n_points; p++) {
            points[p].print(opts);
        }

        int total_count = 0;

        //printf("\nPost-assigned Clusters:\n");
        for(int k = 0; k < opts->n_clusters; k++)
        {
            clusters[k].print(opts);
            total_count += clusters[k].point_count;
        }
        printf("Total Count: %d\n", total_count);
        break;
#endif
    }
    
    timer.stop();

    average_iteration_time /= (opts->max_iter - iter);

    printf("Converged in %d | Total Time: %f | Avg Iter Time: %f | ", opts->max_iter - iter, timer.get_elapsed_time(), average_iteration_time);

    // Free device memory
    cudaFree(d_clusters);
    cudaFree(d_points);


    return;
}