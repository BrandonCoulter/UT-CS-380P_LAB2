#include "kmeans_cuda_shmem.h"

__device__ void add_new_centroid_position(double* shared_cluster_positions, double* point_position, int n_dims)
{
    for(int d = 0; d < n_dims; d++)
    {
        atomicAdd(&shared_cluster_positions[d], point_position[d]);
    }
};

__device__ void find_new_centroid_position(double* shared_cluster_positions, int point_count, int n_dims)
{
    for(int d = 0; d < n_dims; d++)
    {
        shared_cluster_positions[d] = shared_cluster_positions[d] / point_count;
    }
};

__global__ void cuda_shmem::assign_points_to_clusters(struct Centroid* clusters, struct Point* points, struct options_t* opts, int* converged)
{
    extern __shared__ struct Centroid shared_clusters[];

    int gindex = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the index assigned to the specific thread
    int lindex = threadIdx.x; 

    if(lindex < opts->n_clusters)
    {
        shared_clusters[threadIdx.x] = clusters[threadIdx.x];
    }

    __syncthreads(); // Synchronize threads after loading shared memory

    int newClusterID = -999;

    if (gindex < opts->n_points)
    {
        // Declare variables for assignment tracking
        double dis, e_dis, min_squared_dis;
        int originalClusterID = points[gindex].clusterID; // Store the original ClusterID for the assigned centroid
        newClusterID = originalClusterID; // New ClusterID to be updated with closest centroid
        points[gindex].min_distance = DBL_MAX; // Reset the min distance before cluster checking iteration

        // Iterate each cluster and find the closest one based on euclidean_distance
        for (int k = 0; k < opts->n_clusters; k++) {
            dis = 0.0; // Reset distance
            e_dis = 0.0; // Reset euclidean distance

            // Iterate dimensions and calculate sum of squared differences
            squared_distance(shared_clusters[k].position, points[gindex].position, &dis, opts->n_dims);

            // Check if distance from point to cluster k is smaller than the min_distance
            e_dis = sqrt(dis); // Set euclidean distance
            if (e_dis < points[gindex].min_distance) {
                points[gindex].min_distance = e_dis; // Set min_distance to euclidean_distance from point to centroid
                newClusterID = k;                    // Set the new cluster ID
                min_squared_dis = dis;
            }
        }

        // If the cluster ID has changed, mark the point as moved
        if (originalClusterID != newClusterID) {
            // Update the cluster ID for the point
            points[gindex].clusterID = newClusterID;
            // Mark that the clusters haven't converged yet
            atomicCAS(converged, 1, 0);
        }

        // Accumulate cluster stats
        atomicAdd(&shared_clusters[newClusterID].local_sum_squared_diff, min_squared_dis);
        atomicAdd(&shared_clusters[newClusterID].point_count, 1);
        add_new_centroid_position(shared_clusters[newClusterID].new_position, points[gindex].position, opts->n_dims);


    }

    __syncthreads(); // Synchronize threads before loading global memory

    if(lindex < opts->n_clusters)
    {
        // Atomically write local block shared memory back to global memory
        atomicAdd(&clusters[threadIdx.x].point_count, shared_clusters[threadIdx.x].point_count);
        atomicAdd(&clusters[threadIdx.x].local_sum_squared_diff, shared_clusters[threadIdx.x].local_sum_squared_diff);
    }
    
    __syncthreads();

    return;
}

void cuda_shmem::cuda_kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts)
{
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

        //printf("\n\nIteration %d\n", opts->max_iter - iter);
        
        // Set the convergence variable to true, if all conditions met, the loop will closes
        converged = 1;

        // Update previous SSD and reset SSD
        previous_sum_squared_difference = sum_squared_difference;
        sum_squared_difference = 0.0;
        
        // Assign points to clusters based on euclidean distance
        assign_points_to_clusters<<<blocks_per_grid, threads_per_block, cluster_shmem_size>>>(d_clusters, d_points, opts, &converged);
        CUDA_CHECK_ERROR();

        cudaDeviceSynchronize(); // Make sure that all kernals are completed
        
        // Add position of point to cluster
        //add_new_centroid_position<<<blocks_per_grid, threads_per_block, cluster_shmem_size>>>(d_clusters, d_points, opts);
        //CUDA_CHECK_ERROR();

        //cudaDeviceSynchronize(); // Make sure that all kernals are completed



        // Copy device to host memory for clusters and points
        cudaMemcpy(clusters, d_clusters, opts->n_clusters * sizeof(struct Centroid), cudaMemcpyDeviceToHost);
        cudaMemcpy(points, d_points, opts->n_points * sizeof(struct Point), cudaMemcpyDeviceToHost);





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
    

    printf("converged in %d | ", opts->max_iter - iter);

    // Free device memory
    cudaFree(d_clusters);
    cudaFree(d_points);


    return;
}