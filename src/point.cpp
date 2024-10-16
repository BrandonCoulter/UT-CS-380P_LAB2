#include "point.h"

Point::Point(struct options_t opts){
    // std::cout << "Created a point" << std::endl;
    position = (double*)malloc(opts.n_dims * sizeof(double));
    pointID = -1;
    clusterID = -1;
    min_distance = __DBL_MAX__;
}

// Calculate the squared distance between two points with n dimensions 
double Point::squared_distance(double* pos1, double* pos2, struct options_t* opts)
{
    double dis = 0.0;

    for(int d = 0; d < opts->n_dims; d++)
    {
        dis += (pos1[d] - pos2[d]) * (pos1[d] - pos2[d]);
    }
    return dis;
}

// Calculate the euclidean distance between two points with n dimensions 
double Point::euclidean_distance(double* pos1, double* pos2, struct options_t* opts)
{
    return sqrt(squared_distance(pos1, pos2, opts));
}

// Print point data
void Point::print(struct options_t* opts)
{
    printf("Pos: ");
    for(int d = 0; d < opts->n_dims; d++)
    {
        printf(" %f, ", position[d]);
    }

    printf("\nPID: %d\n", pointID);
    printf("CID: %d\n", clusterID);
    printf("Min_Dis: %e\n", min_distance);

    return;
}