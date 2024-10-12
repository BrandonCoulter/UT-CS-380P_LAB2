#include <kmeans.h>

void kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts)
{
    // Initialize kmeans variables
    bool converaged = false;
    int iter = opts->max_iter;
    double previous_sum_squared_difference= 0.0;
    double sum_squared_difference = 0.0;

    // Converagance or max iterations prevent infinite looping
    // Decrease iteration to stop loop incase it doesn't converage
    while(!converaged && --iter)
    {
        // std::cout << iter << std::endl;
        converaged = true;

        // Update previous SSD and reset SSD
        previous_sum_squared_difference = sum_squared_difference;
        sum_squared_difference = 0.0;

        // Assign points to clusters (ID) based on euclidean distance
        for(int p = 0; p < opts->n_points; p++)
        {
            int clusterID = -999; // Reset Cluster ID
        
            // If a point already has a cluster ID associated then assign the checker variable to that ID
            if (points[p].clusterID != -1)
            {
                clusterID = points[p].clusterID;
            }
            // For each cluster check the euclidean distance from the cluster to the point
            for(int k = 0; k < opts->n_clusters; k++)
            {
                // If distance is smaller than the smallest distance, set that to min_distance.
                // This is the closest cluster and therefore the clusterID for that point is set to the cluster's ID
                if(points[p].euclidean_distance(points[p].position, clusters[k].position, opts) < points[p].min_distance)
                {
                    points[p].min_distance = points[p].euclidean_distance(points[p].position, clusters[k].position, opts);
                    clusterID = clusters[k].clusterID;
                }
            }
            // Add each euclidean distance from point to assigned centroid. (WCSS)
            clusters[clusterID].local_sum_squared_diff += points[p].euclidean_distance(points[p].position, clusters[clusterID].position, opts);

            // Set the clusterID and Reset the Min Distance
            points[p].clusterID = clusterID; // Set new clusterID to closest cluster
            points[p].min_distance = __DBL_MAX__; // Reset min_distance for new round 

            // Count the number of points belonging to a cluster.
            // This should be reset every iteration //TODO: RESET POINT COUNT
            clusters[clusterID].point_count += 1;
            clusters[clusterID].add_position(points[p], opts); // Add position to recalculate the mean position after all points assigned

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
            else
            {
                std::cout << "Centroid #" << k << " has no associated points!" << std::endl;
            }

            // Caluclate Sum of differences and set new centroid positions
            sum_squared_difference += clusters[k].local_sum_squared_diff;
            clusters[k].iterate_cluster(opts);

        }

        if(abs(sum_squared_difference - previous_sum_squared_difference) > opts->threshold)
        {
            converaged = false;
        }

#ifdef __PRINT__
        std::cout << "Iteration: " << opts->max_iter - iter << " Inertia = " << abs(sum_squared_difference - previous_sum_squared_difference) << std::endl;
#endif

    }

#ifdef __PRINT__
    // iter += 1; // This is done because the final check of the while loop isn't a loop, it just closes it
    std::cout << "CONVERGED IN " << opts->max_iter - iter << " LOOPS." << std::endl;
#endif

}