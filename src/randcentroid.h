#ifndef _RANDCENTROID_H
#define _RANDCENTROID_H

#include <iostream>
#include <algorithm>

#include "centroid.h"
#include "argparse.h"

int kmeans_rand();

void kmeans_srand(unsigned int);

struct Centroid* gen_initial_centroid(struct options_t *opts, struct Point* points);


#endif