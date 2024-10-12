#include <point.h>

Point::Point(struct options_t opts){
    // std::cout << "Created a point" << std::endl;
    position = (double*)malloc(opts.n_dims * sizeof(double));
    pointID = -1;
    clusterID = -1;
    min_distance = __DBL_MAX__;
}
// Point::~Point()
// {
//     free(position);
// }

double Point::squared_distance(double* pos1, double* pos2, struct options_t* opts)
{
    double dis = 0.0;

    for(int d = 0; d < opts->n_dims; d++)
    {
        dis += (pos1[d] - pos2[d]) * (pos1[d] - pos2[d]);
    }
    return dis;
}

double Point::euclidean_distance(double* pos1, double* pos2, struct options_t* opts)
{
    return sqrt(squared_distance(pos1, pos2, opts));
}

void Point::print(struct options_t* opts)
{
    std::cout << "Pos: ";
    for(int d = 0; d < opts->n_dims; d++)
    {
        std::cout << " " << position[d] << ", ";
    }
    std::cout << "\nPID: " << pointID << std::endl;
    std::cout << "CID: " << clusterID << std::endl;
    std::cout << "Min_Dis: " << min_distance << std::endl;

    return;
}