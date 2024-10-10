#include <io.h>

int get_n_points(struct options_t* opts)
{
  	// Open file
	std::ifstream in;
	in.open(opts->in_file);
    std::string line;
    getline(in, line);
    return stoi(line);
}

void read_file(struct options_t* opts, struct Point* points)
{

  	// Open file
	std::ifstream in;
	in.open(opts->in_file);
	
    int d = 0;
    std::string line;
    std::string dim_pos;
    getline(in, line); // Drop the first line which corresponds to num points

    // std::cout << "IO N_POINTS: " << opts->n_points << std::endl;
    for(int i = 0; i < opts->n_points; ++i)
    {
        getline(in, line);
        std::stringstream ss(line);
        d = -1;
        
        struct Point point(*opts);

        while(getline(ss, dim_pos, ' '))
        {
            if(d >= opts->n_dims + 1)
            {
                break;
            }
            else if (d != -1 and d < opts->n_dims)
            {
                point.position[d] = stod(dim_pos);
                // std::cout << " " << stold(dim_pos) << " ";
            }
            else if (d == -1){
                point.pointID = stoi(dim_pos);
            }

            d++;
        }
        // point.print(opts);
        points[i] = point;
        // std::cout << std::endl;
    }
    // std::cout << std::endl;
    in.close();
}

// void write_file(struct options_t*         args,
//                	struct prefix_sum_args_t* opts) {
//   // Open file
// 	std::ofstream out;
// 	out.open(args->out_file, std::ofstream::trunc);

// 	// Write solution to output file
// 	for (int i = 0; i < opts->n_vals; ++i) {
// 		out << opts->output_vals[i] << std::endl;
// 	}

// 	out.flush();
// 	out.close();
	
// 	// Free memory
// 	free(opts->input_vals);
// 	free(opts->output_vals);
// }
