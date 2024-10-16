// #include <iostream>
#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        printf("Usage:\n");
        printf("\t-d <num_dimensions>(int)\n");
        printf("\t-i <input_file_path>(*char)\n");
        printf("\t-m <max_num_iteration>(int)\n");
        printf("\t-t <threshold>(double)\n");
        printf("\t-s <seed>(int)\n");
        printf("\t[Optional] -T <print_time>(bool)\n");
        printf("\t[Optional] -c <print_cent>(bool)\n");
        printf("\t[Optional] --run_cuda <run with cuda basic>(bool)\n");
        printf("\t[Optional] --run_shmem <run with cuda shmem>(bool)\n");
        printf("\t[Optional] --run_thrust <run with thrust>(bool)\n");
        exit(0);
    }

    opts->print_cent = false;
    opts->print_time = false;
    opts->run_cuda = false;
    opts->run_shmem = false;
    opts->run_thrust = false;

    struct option l_opts[] = {
        {"k", required_argument, NULL, 'k'},
        {"d", required_argument, NULL, 'd'},
        {"i", required_argument, NULL, 'i'},
        {"m", required_argument, NULL, 'm'},
        {"t", required_argument, NULL, 't'},
        {"s", required_argument, NULL, 's'},
        {"c", no_argument, NULL, 'c'},
        {"T", no_argument, NULL, 'T'},
        {"run_cuda", no_argument, NULL, 'x'},
        {"run_shmem", no_argument, NULL, 'y'},
        {"run_thrust", no_argument, NULL, 'z'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:cxyzT", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->n_clusters = atoi((char *)optarg);
            break;
        case 'd':
            opts->n_dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'm':
            opts->max_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'c':
            opts->print_cent = true;
            break;
        case 'T':
            opts->print_time = true;
            break;
        case 'x':
            opts->run_cuda = true;
            break;
        case 'y':
            opts->run_shmem = true;
            break;
        case 'z':
            opts->run_thrust = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(0);
        }
    }
}
