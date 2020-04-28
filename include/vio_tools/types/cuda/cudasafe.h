/**
 * @file cudasafe.h
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */
#ifndef SRC_CUDASAFE_H
#define SRC_CUDASAFE_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace vio_tools
{
    namespace cuda
    {
        inline void cuda_assert(cudaError_t code, const char *file, int line, bool abort = true)
        {
            if (code != cudaSuccess)
            {
                fprintf(stderr, "cuda_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
            }
        }
    }
}
#endif //SRC_CUDASAFE_H
