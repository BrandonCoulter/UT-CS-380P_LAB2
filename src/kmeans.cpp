#include <kmeans.h>

void kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts)
{

    bool converaged = false;
    int iter = opts->max_iter;

    // TEST CODE: REMOVE
    // struct Point test_point(*opts);
    // for(int d = 0; d < opts->n_dims; d++){
    //     test_point.position[d] = 0;
    // }
    // test_point.position = points[0].position;


    // Converagance or max iterations prevent infinite looping
    // Decrease iteration to stop loop incase it doesn't converage
    while(!converaged && iter--)
    {
        // std::cout << iter << std::endl;
        converaged = true;

        int clusterID = -999;
        // struct Point* new_clusters = (struct Point*) malloc(opts->n_clusters*sizeof(struct Point));

        // Assign points to clusters (ID) based on euclidean distance
        for(int p = 0; p < opts->n_points; p++)
        {

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
                if(points[p].distance(clusters[k], opts) < points[p].min_distance)
                {
                    points[p].min_distance = points[p].distance(clusters[k], opts);
                    clusterID = clusters[k].clusterID;
                }
            }

            // If every point did not change clusterIDs then convergance has happened
            // If the clusterID is different then set the converaged to false continue loop again
            if (points[p].clusterID != clusterID)
            {
                converaged = false;
            }

            // Set the clusterID and Reset the Min Distance
            points[p].clusterID = clusterID; // Set new clusterID to closest cluster
            points[p].min_distance = __DBL_MAX__; // Reset min_distance for new round 
            // Count the number of points belonging to a cluster.
            // This should be reset every iteration //TODO: RESET POINT COUNT
            clusters[clusterID].point_count += 1;
            clusters[clusterID].add_position(points[p], opts); // Add position to recalculate the mean position after all points assigned

            // std::cout << "POINTS: " << std::endl;
            // points[p].print(opts);

        }

        // Iterate the clusters and recompute the centroids
        for(int k = 0; k < opts->n_clusters; k++)
        {
            // Check if the distance from the old position to the new position
            // is less than some given threshold t (opts.threshold)
            clusters[k].find_new_center(opts); // First compute the new centroid
            clusters[k].point_count = 0;

            if(!clusters[k].threshold_check(opts))
            {
                converaged = false;
            }
            // std::cout << "POST-CLUSTER: " << std::endl;
            // clusters[k].print(opts);

        }

        // Recompute Centroids

        // TEST CODE: REMOVE
        // for(int p = 0; p < opts->n_points; p++)
        // {
        //     if(points[p].clusterID == 0){
        //         points_belonging_to_cluter += 1;
        //         test_point.add_position(points[p], opts);
        //     }
        // }
        // test_point.find_mean(points_belonging_to_cluter, opts);
        // test_point.print(opts);
        // break;

    }

#ifdef __PRINT__
    iter += 1; // This is done because the final check of the while loop isn't a loop, it just closes the loop
    std::cout << "CONVERGED IN " << opts->max_iter - iter << " LOOPS." << std::endl;
#endif

}