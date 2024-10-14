#ifndef _POINT_H
#define _POINT_H

#include <math.h>

#include "argparse.h"

struct Point
{
    double*        position;
    double         min_distance;
    int            pointID;
    int            clusterID;

    Point(struct options_t opts);
    // ~Point();
    double squared_distance(double* pos1, double* pos2, struct options_t* opts);
    double euclidean_distance(double* pos1, double* pos2, struct options_t* opts);
    void print(struct options_t* opts);
};
#endif