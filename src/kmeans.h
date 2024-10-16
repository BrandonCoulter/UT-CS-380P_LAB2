#ifndef _KMEANS_H
#define _KMEANS_H

#include <chrono>

#include "point.h"
#include "centroid.h"
#include "argparse.h"

bool is_converged();
void kmeans(struct Centroid* clusters, struct Point* points, struct options_t* opts);

#endif
