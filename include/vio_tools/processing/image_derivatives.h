/**
 * @file image_derivatives.h
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */
#ifndef SRC_IMAGE_DERIVATIVES_H
#define SRC_IMAGE_DERIVATIVES_H

#include <vio_tools/types/image.h>
#include <cstdint>

namespace vio_tools
{
    void sobel(const Image<uint8_t> &I, Image<int16_t> &Ix, Image<int16_t> &Iy);

    Image<int16_t> sobel_x(const Image<uint8_t> &I);

    Image<int16_t> sobel_y(const Image<uint8_t> &I);
}
#endif //SRC_IMAGE_DERIVATIVES_H
