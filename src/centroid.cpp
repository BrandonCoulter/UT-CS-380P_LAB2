#include <centroid.h>

// Constructor for a Centroid - Inherits Point
Centroid::Centroid(struct options_t opts) : Point(opts)
{
    new_position = (double*)malloc(opts.n_dims * sizeof(double));
    point_count = 0;
    is_cluster = true;
}

// Overides the Point print method to print extra centroid data
void Centroid::print(struct options_t* opts)
{
    std::cout << "Pos: ";
    for(int d = 0; d < opts->n_dims; d++)
    {
        std::cout << " " << position[d] << ", ";
    }
    std::cout << "\nNew Pos: ";
    for(int d = 0; d < opts->n_dims; d++)
    {
        std::cout << " " << new_position[d] << ", ";
    }
    std::cout << "\nPID: " << pointID << std::endl;
    std::cout << "CID: " << clusterID << std::endl;
    std::cout << "Min_Dis: " << min_distance << std::endl;
    std::cout << "P_Count: " << point_count << std::endl;

    return;
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

    double weight = 0.1;

    // If a cluster has 1 or more points assigned to it
    // find the mean and set that to the new position for that dimension
    for(int d = 0; d < opts->n_dims; d++)
    {
        new_position[d] = (weight * position[d]) + ((1 - weight) * (new_position[d] / point_count));
    }
    return;
}

// Check the distance moved by the centroid against a predefined threshold
// If the distance is less than the threshold then convergance has occured
// bool Centroid::threshold_check(struct options_t* opts)
// {
//     bool converaged = true;

//     if(point_count <=0)
//     {
//         return converaged;
//     }

//     // For each dimension check the old position vs new position
//     for(int d = 0; d < opts->n_dims; d++)
//     {
//         // Check if the distance is less than the threshold
//         if(abs(position[d] - new_position[d]) > opts->threshold)
//         {
//             converaged = false; // If any dimension is more than threshold convergance is false
//         }
//         position[d] = new_position[d]; // Update position to new position
//         new_position[d] = 0; // Reset new position for next iteration
//     }
    
//     return converaged;
// }
bool Centroid::threshold_check(struct options_t* opts)
{
    double dis = 0.0;

    if(point_count <=0)
    {
        return true;
    }

    for(int d = 0; d < opts->n_dims; d++)
    {
        dis += (new_position[d] - position[d]);
        position[d] = new_position[d]; // Update position to new position
        new_position[d] = 0; // Reset new position for next iteration
    }
    
    if(dis < opts->threshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}