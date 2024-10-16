#include "kmeans_cuda_basic.h"


// Control the assignment of points to clusters via euclidean distance
__global__ void cuda_basic::assign_points_to_clusters(struct Centroid* clusters, struct Point* points, struct options_t* opts, int* converged) {
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the index assigned to the specific thread

    if (index < opts->n_points) {
        // Declare variables for assignment tracking
        double dis, e_dis, min_squared_dis;
        int originalClusterID = points[index].clusterID; // Store the original ClusterID for the assigned centroid
        int newClusterID = originalClusterID; // New ClusterID to be updated with closest centroid
        points[index].min_distance = DBL_MAX; // Reset the min distance before cluster checking iteration

        // Iterate each cluster and find the closest one based on euclidean_distance
        for (int k = 0; k < opts->n_clusters; k++) {
            dis = 0.0; // Reset distance
            e_dis = 0.0; // Reset euclidean distance

            // Iterate dimensions and calculate sum of squared differences
            squared_distance(clusters[k].position, points[index].position, &dis, opts->n_dims);

            // Check if distance from point to cluster k is smaller than the min_distance
            e_dis = sqrt(dis); // Set euclidean distance
            if (e_dis < points[index].min_distance) {
                points[index].min_distance = e_dis; // Set min_distance to euclidean_distance from point to centroid
                newClusterID = k;                  // Set the new cluster ID
                min_squared_dis = dis;
            }
        }

        // If the cluster ID has changed, mark the point as moved
        if (originalClusterID != newClusterID) {
            // Update the cluster ID for the point
            points[index].clusterID = newClusterID;
            // Mark that the clusters haven't converged yet
            atomicCAS(converged, 1, 0);
        }

        // Accumulate cluster stats
        atomicAdd(&clusters[newClusterID].local_sum_squared_diff, min_squared_dis);
        atomicAdd(&clusters[newClusterID].point_count, 1);
    }
}


__global__ void cuda_basic::add_new_centroid_position(struct Centroid* clusters, struct Point* points, struct options_t* opts)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the index assigned to the specific thread
 
    if (index < opts->n_points) {

        for(int d = 0; d < opts->n_dims; d++)
        {
            atomicAdd(&clusters[points[index].clusterID].new_position[d], points[index].position[d]);
            //clusters[points[index].clusterID].new_position[d] += points[index].position[d];
        }

    }
}

__global__ void cuda_basic::find_new_centroid_position(struct Centroid* clusters, struct Point* points, struct options_t* opts)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the index assigned to the specific thread
 
    if (index < opts->n_clusters) {
    
        for(int d = 0; d < opts->n_dims; d++)
        {
            clusters[index].new_position[d] = clusters[index].new_position[d] / clusters[index].point_count;
        }

    }
}

__global__ void cuda_basic::reset_centroids(struct Centroid* clusters, struct options_t* opts)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the index assigned to the specific thread
 
    if (index < opts->n_points) {
    
        for(int d = 0; d < opts->n_dims; d++)
        {

            // Set the position to the new centroid position and reset new position
            clusters[index].position[d] = clusters[index].new_position[d];
            clusters[index].new_position[d] = 0;
        }

        // Reset local SSD and point count
        clusters[index].local_sum_squared_diff = 0.0;
        clusters[index].point_count = 0.0;

    }
}

void cuda_basic::cuda_kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts) {
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
        assign_points_to_clusters<<<blocks_per_grid, threads_per_block>>>(d_clusters, d_points, opts, &converged);
        CUDA_CHECK_ERROR();

        cudaDeviceSynchronize(); // Make sure that all kernals are completed

        // Add position of point to cluster
        add_new_centroid_position<<<blocks_per_grid, threads_per_block>>>(d_clusters, d_points, opts);
        CUDA_CHECK_ERROR();

        cudaDeviceSynchronize(); // Make sure that all kernals are completed
        
        // Find the mean after all points have been added
        find_new_centroid_position<<<1, opts->n_clusters>>>(d_clusters, d_points, opts);
        //CUDA_CHECK_ERROR();

        cudaDeviceSynchronize(); // Make sure that all kernals are completed

        // Copy device to host memory for clusters and points
        cudaMemcpy(clusters, d_clusters, opts->n_clusters * sizeof(struct Centroid), cudaMemcpyDeviceToHost);
        cudaMemcpy(points, d_points, opts->n_points * sizeof(struct Point), cudaMemcpyDeviceToHost);
        

        for(int k = 0; k < opts->n_clusters; k++)
        {
            // Caluclate Sum of differences and set new centroid positions
            sum_squared_difference += clusters[k].local_sum_squared_diff;
        }

        if(abs(sum_squared_difference - previous_sum_squared_difference) > opts->threshold)
        {
            converged = 0; // set convergence to false
        }

        // Reset all the centroids for the next iteration
        reset_centroids<<<1, opts->n_clusters>>>(d_clusters, opts);
        //CUDA_CHECK_ERROR();
        cudaDeviceSynchronize(); // Make sure that all kernals are completed

#ifdef __PRINT__

        // DEBUG PRINTING

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
}
