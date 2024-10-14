#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {

    int n_points;
    int n_clusters;
    int n_dims;
    char *in_file;
    int max_iter;
    double threshold;
    bool print_cent;
    int seed;
    bool print_time;
    bool run_cuda;
    bool run_shmem;
    bool run_thrust;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif
