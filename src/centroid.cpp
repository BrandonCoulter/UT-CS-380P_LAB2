#include "centroid.h"

// Constructor for a Centroid - Inherits Point
Centroid::Centroid(struct options_t opts) : Point(opts)
{
    new_position = (double*)malloc(opts.n_dims * sizeof(double));
    local_sum_squared_diff = 0.0;
    point_count = 0;

    for(int d = 0; d < opts.n_dims; d++)
    {
        new_position[d] = 0;
    }
}

// Adds Point dimension data together to later find the mean 
void Centroid::add_position(Point p, struct options_t* opts)
{
    for(int d = 0; d < opts->n_dims; d++)
    {
        new_position[d] += p.position[d];
    }
    return;
}
// Finds the mean for each cluster point dimension. 
// Updates the new position to the average of all the points in that cluster
void Centroid::find_new_center(struct options_t* opts)
{
    // If a cluster is orphaned do not update position
    if(point_count <= 0)
    {
        return;
    }

    // If a cluster has 1 or more points assigned to it
    // find the mean and set that to the new position for that dimension
    for(int d = 0; d < opts->n_dims; d++)
    {
        new_position[d] = (new_position[d] / point_count);
    }
    return;
}

// Reset the centroid for the next iteration, set the position to be the new position
void Centroid::iterate_cluster(struct options_t* opts)
{
    local_sum_squared_diff = 0.0; // Reset SSD variable
    point_count = 0; // Reset point count variable

    for(int d = 0; d < opts->n_dims; d++)
    {
        position[d] = new_position[d]; // Update position to new position
        new_position[d] = 0; // Reset new position for next iteration
    }
}

// Overides the Point print method to print extra centroid data
void Centroid::print(struct options_t* opts)
{
    printf("Pos: ");
    for(int d = 0; d < opts->n_dims; d++)
    {
        printf(" %f, ", position[d]);
    }
    printf("\nNew Pos: ");
    for(int d = 0; d < opts->n_dims; d++)
    {
        printf(" %f, ", new_position[d]);
    }
    printf("\nPID: %d\n", pointID);
    printf("CID: %d\n", clusterID);
    printf("Min_Dis: %e\n", min_distance);
    printf("P_Count: %d\n", point_count);
    printf("L_SSD: %f\n", local_sum_squared_diff);

    return;
}