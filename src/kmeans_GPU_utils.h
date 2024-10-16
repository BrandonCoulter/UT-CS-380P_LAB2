#ifndef _KMEANS_GPU_UTILS_H
#define _KMEANS_GPU_UTILS_H

#define CUDA_CHECK_ERROR() {                                          \
    cudaError_t err = cudaGetLastError();                             \
    if (err != cudaSuccess) {                                         \
        printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err),  \
                __FILE__, __LINE__);                                  \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

inline __device__ void squared_distance(double* c_pos, double* p_pos, double* dis, int n_dims)
{
    for(int d = 0; d < n_dims; d++)
    {
        *dis += (c_pos[d] - p_pos[d]) * (c_pos[d] - p_pos[d]);
    }
}

struct CudaTimer
{
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    inline CudaTimer()
    {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    inline ~CudaTimer()
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    inline void start()
    {
        cudaEventRecord(start_event);
    }
    inline void stop()
    {
        cudaEventRecord(stop_event);
    }
    inline float get_elapsed_time()
    {
        cudaEventSynchronize(stop_event);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);

        return milliseconds;
        
    }

};

#endif