#ifndef _CENTROID_H
#define _CENTROID_H

#include <point.h>


struct Centroid : Point
{
    double*   new_position;
    int            point_count;
    bool           is_cluster;

    Centroid(struct options_t opts);
    void print(struct options_t* opts);
    void add_position(Point p, struct options_t* opts);
    void find_new_center(struct options_t* opts);
    bool threshold_check(struct options_t* opts);
};
#endif