/**
 * @file image_derivatives.cpp
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */
#include "vio_tools/processing/image_derivatives.h"

void vio_tools::sobel(const Image<uint8_t> &I, Image<int16_t> &Ix, Image<int16_t> &Iy)
{
    Ix = sobel_x(I);
    Iy = sobel_y(I);
}

vio_tools::Image<int16_t> vio_tools::sobel_x(const Image<uint8_t> &I)
{
    Image<int16_t> I0(I.rows(), I.cols());
    Image<int16_t> Ix(I.rows(), I.cols());

    for (int i = 0; i < I.rows(); ++i)
        for (int j = 0; j < I.cols(); ++j)
            I0(i, j) = I(i, j + 1) - I(i, j - 1);

    for (int i = 0; i < I.rows(); ++i)
        for (int j = 0; j < I.cols(); ++j)
            Ix(i, j) = I0(i - 1, j) + 2 * I0(i, j) + I0(i + 1, j);

    return Ix;
}

vio_tools::Image<int16_t> vio_tools::sobel_y(const Image<uint8_t> &I)
{
    Image<int16_t> I0(I.rows(), I.cols());
    Image<int16_t> Iy(I.rows(), I.cols());

    for (int i = 0; i < I.rows(); ++i)
        for (int j = 0; j < I.cols(); ++j)
            I0(i, j) = I(i + 1, j) - I(i - 1, j);

    for (int i = 0; i < I.rows(); ++i)
        for (int j = 0; j < I.cols(); ++j)
            Iy(i, j) = I0(i, j - 1) + 2 * I0(i, j) + I0(i, j + 1);

    return Iy;
}