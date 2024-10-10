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

double Point::distance(Point p, struct options_t* opts)
{
    double dis = 0.0;

    for(int d = 0; d < opts->n_dims; d++)
    {
        dis += (position[d] - p.position[d]) * (position[d] - p.position[d]);
    }
    return sqrt(dis);
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