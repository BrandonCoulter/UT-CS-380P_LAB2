#include <centroid.h>



Centroid::Centroid(struct options_t opts) : Point(opts)
{
    new_position = (double*)malloc(opts.n_dims * sizeof(double));
    point_count = 0;
    is_cluster = true;
}

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

void Centroid::add_position(Point p, struct options_t* opts)
{
    for(int d = 0; d < opts->n_dims; d++)
    {
        new_position[d] += p.position[d];
        // std::cout << "NEW POS[d]: " << new_position[d] << " | POINT POS[d]: " << p.position[d] << std::endl;
    }
    return;
}

void Centroid::find_new_center(struct options_t* opts)
{
    if(point_count <= 0)
    {
        return;
    }

    for(int d = 0; d < opts->n_dims; d++)
    {
        new_position[d] = new_position[d] / point_count;
    }
    return;
}

bool Centroid::threshold_check(struct options_t* opts)
{
    double dis = 1.0;

    for(int d = 0; d < opts->n_dims; d++)
    {
        dis += (position[d] - new_position[d]) * (position[d] - new_position[d]);
        position[d] = new_position[d];
        new_position[d] = 0;
    }
    
    // print(opts);

    if(sqrt(dis) < opts->threshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}