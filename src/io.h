#ifndef _IO_H
#define _IO_H

#include <iostream>
#include <fstream>
#include <sstream>

#include "point.h"
#include "argparse.h"

int get_n_points(struct options_t* opts);

void read_file(struct options_t* opts, struct Point* points);

#endif