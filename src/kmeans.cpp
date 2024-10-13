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
                // std::cout << "Distance: " << distance << std::endl;
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
                std::cout << "Point #" << points[p].pointID << " changed clusters IDs from #" << points[p].clusterID << " to Cluster #" << clusterID << std::endl; 
#endif
                converaged = false;
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
            else
            {
                std::cout << "Centroid #" << k << " has no associated points!" << std::endl;
            }

            // Caluclate Sum of differences and set new centroid positions
            sum_squared_difference += clusters[k].local_sum_squared_diff;
            
            if(clusters[k].squared_distance(clusters[k].position, clusters[k].new_position, opts) > opts->threshold)
            {
#if defined(__PRINT__) && defined(__VERBOSE__)       
                std::cout << "Cluster #" << k << " moved more than the threshold." << std::endl; 
#endif
                converaged = false;
            }
            clusters[k].iterate_cluster(opts);

        }

        if(abs(sum_squared_difference - previous_sum_squared_difference) > opts->threshold)
        {
            converaged = false;
        }

#if defined(__PRINT__) && defined(__VERBOSE__)
        std::cout << "Iteration: " << opts->max_iter - iter << " Inertia = " << abs(sum_squared_difference - previous_sum_squared_difference) << "\n" <<std::endl;
#endif

    }

#ifdef __PRINT__
    // iter += 1; // This is done because the final check of the while loop isn't a loop, it just closes it
    std::cout << "CONVERGED IN " << opts->max_iter - iter << " LOOPS." << std::endl;
#endif

}