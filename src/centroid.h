#ifndef _CENTROID_H
#define _CENTROID_H

#include "point.h"


struct Centroid : Point
{
    double*        new_position;
    double         local_sum_squared_diff;
    int            point_count;

    Centroid(struct options_t opts);
    void print(struct options_t* opts);
    void add_position(Point p, struct options_t* opts);
    void find_new_center(struct options_t* opts);
    void iterate_cluster(struct options_t* opts);
};
#endif