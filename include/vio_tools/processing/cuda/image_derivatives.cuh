/**
 * @file cuda_image_derivatives.h
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */
#ifndef SRC_IMAGE_DERIVATIVES_CUH
#define SRC_IMAGE_DERIVATIVES_CUH

#include <vio_tools/types/image.h>
#include <stdint.h>

namespace vio_tools
{
    namespace cuda
    {
        void sobel(const Image<uint8_t> &I, Image<int16_t> &Ix, Image<int16_t> &Iy);

        Image<int16_t> sobel_x(const Image<uint8_t> &I);

        Image<int16_t> sobel_y(const Image<uint8_t> &I);
    }

    namespace kernel
    {
        __global__ void sobel_x_conv1(cudaTextureObject_t I, cudaSurfaceObject_t Ix, int rows, int cols);

        __global__ void sobel_x_conv2(cudaTextureObject_t I, cudaSurfaceObject_t Ix, int rows, int cols);

        __global__ void sobel_y_conv1(cudaTextureObject_t I, cudaSurfaceObject_t Iy, int rows, int cols);

        __global__ void sobel_y_conv2(cudaTextureObject_t I, cudaSurfaceObject_t Iy, int rows, int cols);
    }
}
#endif //SRC_IMAGE_DERIVATIVES_CUH
