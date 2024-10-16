#include "kmeans.h"

void kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts)
{
    // Timer metric variables
    double total_time = 0;
    double average_iteration_time = 0;
    // Start total sequential timer
    auto total_start = std::chrono::high_resolution_clock::now();

    // Initialize kmeans variables
    bool converged = false;
    int iter = opts->max_iter;
    double previous_sum_squared_difference= 0.0;
    double sum_squared_difference = 0.0;

    // Converagance or max iterations prevent infinite looping
    // Decrease iteration to stop loop incase it doesn't converage
    while(!converged && --iter)
    {

        converged = true;

        // Update previous SSD and reset SSD
        previous_sum_squared_difference = sum_squared_difference;
        sum_squared_difference = 0.0;

        // Assign points to clusters (ID) based on euclidean distance
        for(int p = 0; p < opts->n_points; p++)
        {
            int clusterID = -999; // Reset Cluster ID checking variable
        
            // If a point already has a cluster ID associated then assign the checker variable to that ID
            if (points[p].clusterID != -1)
            {
                clusterID = points[p].clusterID;
            }
            // For each cluster check the euclidean distance from the cluster to the point
            for(int k = 0; k < opts->n_clusters; k++)
            {
                // If distance is smaller than the smallest distance, set that to min_distance.
                // This is the closest cluster and therefore the associated clusterID for that point is set to the cluster's ID
                double distance = points[p].euclidean_distance(points[p].position, clusters[k].position, opts);
                if(distance < points[p].min_distance)
                {
                    points[p].min_distance = distance;
                    clusterID = clusters[k].clusterID;
                }
            }
            

            // If any point changed cluster IDs then convergence is false
            if(points[p].clusterID != clusterID)
            {
#if defined(__PRINT__) && defined(__VERBOSE__)
                printf("Point # %d changed cluster IDs from #%d to Cluster #%d\n", points[p].pointID, points[p].clusterID, clusterID);
#endif
                converged = false;
            }
            
            
            // Add each squared distance from point to assigned centroid to the associated centroids
            // local sum of squares. (WCSS)
            clusters[clusterID].local_sum_squared_diff += points[p].squared_distance(points[p].position, clusters[clusterID].position, opts);

            // Set the clusterID and Reset the Min Distance
            points[p].clusterID = clusterID; // Set new clusterID to closest cluster            
            points[p].min_distance = __DBL_MAX__; // Reset min_distance for new round 

            // Count the number of points belonging to a cluster.
            clusters[clusterID].point_count += 1;
            // Add position to recalculate the mean position after all points assigned
            clusters[clusterID].add_position(points[p], opts); 

        }

        // Iterate the clusters and recompute the centroids
        for(int k = 0; k < opts->n_clusters; k++)
        {
            // Check if the distance from the old position to the new position
            // is less than some given threshold t (opts.threshold)
            if (!(clusters[k].point_count <= 0))
            {
                clusters[k].find_new_center(opts); // First compute the new centroid
            }

            // Caluclate Sum of differences and set new centroid positions
            sum_squared_difference += clusters[k].local_sum_squared_diff;
            
            clusters[k].iterate_cluster(opts);

        }

        if(abs(sum_squared_difference - previous_sum_squared_difference) > opts->threshold)
        {
            converged = false;
        }

#if defined(__PRINT__) && defined(__VERBOSE__)
        printf("Iteration %d Inertia = %e\n", opts->max_iter - iter, abs(sum_squared_difference - previous_sum_squared_difference));
#endif

    }

    //End timer and print out elapsed
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time_clock = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    total_time = total_time_clock.count();
    average_iteration_time = total_time / (opts->max_iter - iter);

    printf("Converged in %d | Total Time: %f | Avg Iter Time: %f | ", opts->max_iter - iter, total_time, average_iteration_time);

}