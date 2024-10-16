#include "io.h"

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
            }
            else if (d == -1){
                point.pointID = stoi(dim_pos);
            }

            d++;
        }

        points[i] = point;
    }
    in.close();
}
