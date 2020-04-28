/**
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */
#include "vio_tools/processing/cuda/image_derivatives.cuh"

#include <vio_tools/types/cuda/surf_tex.h>

#include <surface_indirect_functions.h>

__global__ void vio_tools::kernel::sobel_x_conv1(cudaTextureObject_t I, cudaSurfaceObject_t Ix, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int16_t dx = (int16_t) tex2D<uint8_t>(I, x + 1, y) - (int16_t) tex2D<uint8_t>(I, x - 1, y);
        surf2Dwrite(dx, Ix, x * sizeof(int16_t), y);
    }
}

__global__ void vio_tools::kernel::sobel_x_conv2(cudaTextureObject_t I, cudaSurfaceObject_t Ix, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int16_t s = tex2D<int16_t>(I, x, y - 1) + 2 * tex2D<int16_t>(I, x, y) + tex2D<int16_t>(I, x, y + 1);
        surf2Dwrite(s, Ix, x * sizeof(int16_t), y);
    }
}

__global__ void vio_tools::kernel::sobel_y_conv1(cudaTextureObject_t I, cudaSurfaceObject_t Iy, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int16_t dy = (int16_t) tex2D<uint8_t>(I, x, y + 1) - (int16_t) tex2D<uint8_t>(I, x, y - 1);
        surf2Dwrite(dy, Iy, x * sizeof(int16_t), y);
    }
}

__global__ void vio_tools::kernel::sobel_y_conv2(cudaTextureObject_t I, cudaSurfaceObject_t Iy, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int16_t s = tex2D<int16_t>(I, x - 1, y) + 2 * tex2D<int16_t>(I, x, y) + tex2D<int16_t>(I, x + 1, y);
        surf2Dwrite(s, Iy, x * sizeof(int16_t), y);
    }
}

void vio_tools::cuda::sobel(const Image<uint8_t> &I, Image<int16_t> &Ix, Image<int16_t> &Iy)
{
    // Create CUDA surfaces/textures
    SurfTex<uint8_t, cudaChannelFormatKindUnsigned> cuda_I(I.rows(), I.cols());
    SurfTex<int16_t, cudaChannelFormatKindSigned> cuda_Ix1(I.rows(), I.cols());
    SurfTex<int16_t, cudaChannelFormatKindSigned> cuda_Iy1(I.rows(), I.cols());
    SurfTex<int16_t, cudaChannelFormatKindSigned> cuda_Ix2(I.rows(), I.cols());
    SurfTex<int16_t, cudaChannelFormatKindSigned> cuda_Iy2(I.rows(), I.cols());

    cuda_I.upload(I.data());

    // CUDA processing
    dim3 dim_block(32, 32);
    dim3 dim_grid((I.cols() + dim_block.x - 1) / dim_block.x, (I.rows() + dim_block.y - 1) / dim_block.y);
    kernel::sobel_x_conv1<<<dim_grid, dim_block>>>(cuda_I.tex_obj(), cuda_Ix1.surf_obj(), I.rows(), I.cols());
    kernel::sobel_y_conv1<<<dim_grid, dim_block>>>(cuda_I.tex_obj(), cuda_Iy1.surf_obj(), I.rows(), I.cols());
    kernel::sobel_x_conv2<<<dim_grid, dim_block>>>(cuda_Ix1.tex_obj(), cuda_Ix2.surf_obj(), I.rows(), I.cols());
    kernel::sobel_y_conv2<<<dim_grid, dim_block>>>(cuda_Iy1.tex_obj(), cuda_Iy2.surf_obj(), I.rows(), I.cols());

    // Download
    cuda_Ix2.download(Ix.data());
    cuda_Iy2.download(Iy.data());
}

vio_tools::Image<int16_t> vio_tools::cuda::sobel_x(const Image<uint8_t> &I)
{
    // Output image object
    Image<int16_t> Ix(I.rows(), I.cols());

    // Create CUDA surfaces/textures
    SurfTex<uint8_t, cudaChannelFormatKindUnsigned> cuda_I(I.rows(), I.cols());
    SurfTex<int16_t, cudaChannelFormatKindSigned> cuda_Ix1(I.rows(), I.cols());
    SurfTex<int16_t, cudaChannelFormatKindSigned> cuda_Ix2(I.rows(), I.cols());

    cuda_I.upload(I.data());

    // CUDA processing
    dim3 dim_block(16, 16);
    dim3 dim_grid((I.cols() + dim_block.x - 1) / dim_block.x, (I.rows() + dim_block.y - 1) / dim_block.y);
    kernel::sobel_x_conv1<<<dim_grid, dim_block>>>(cuda_I.tex_obj(), cuda_Ix1.surf_obj(), I.rows(), I.cols());
    kernel::sobel_x_conv2<<<dim_grid, dim_block>>>(cuda_Ix1.tex_obj(), cuda_Ix2.surf_obj(), I.rows(), I.cols());

    // Download
    cuda_Ix2.download(Ix.data());

    return Ix;
}

vio_tools::Image<int16_t> vio_tools::cuda::sobel_y(const Image<uint8_t> &I)
{
    // Output image object
    Image<int16_t> Iy(I.rows(), I.cols());

    // Create CUDA surfaces/textures
    SurfTex<uint8_t, cudaChannelFormatKindUnsigned> cuda_I(I.rows(), I.cols());
    SurfTex<int16_t, cudaChannelFormatKindSigned> cuda_Iy1(I.rows(), I.cols());
    SurfTex<int16_t, cudaChannelFormatKindSigned> cuda_Iy2(I.rows(), I.cols());

    cuda_I.upload(I.data());

    // CUDA processing
    dim3 dim_block(16, 16);
    dim3 dim_grid((I.cols() + dim_block.x - 1) / dim_block.x, (I.rows() + dim_block.y - 1) / dim_block.y);
    kernel::sobel_y_conv1<<<dim_grid, dim_block>>>(cuda_I.tex_obj(), cuda_Iy1.surf_obj(), I.rows(), I.cols());
    kernel::sobel_y_conv2<<<dim_grid, dim_block>>>(cuda_Iy1.tex_obj(), cuda_Iy2.surf_obj(), I.rows(), I.cols());

    // Download
    cuda_Iy2.download(Iy.data());

    return Iy;
}