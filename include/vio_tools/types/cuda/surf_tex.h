/**
 * @file surf_tex.h
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */
#ifndef SRC_SURF_TEX_H
#define SRC_SURF_TEX_H

#include <vio_tools/types/cuda/cudasafe.h>
#include <cuda_runtime.h>
#include <typeinfo>

namespace vio_tools
{
    namespace cuda
    {
        template<typename T, cudaChannelFormatKind channel_format>
        class SurfTex
        {
        public:
            SurfTex(int rows, int cols)
                    : rows_(rows),
                      cols_(cols)
            {
                // Initialize CUDA array
                cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8 * sizeof(T), 0, 0, 0, channel_format);
                cuda_assert(cudaMallocArray(&cuda_array_, &channel_desc, cols_ * sizeof(T), rows_), __FILE__, __LINE__);

                // Specify surface/texture
                struct cudaResourceDesc res_desc;
                std::memset(&res_desc, 0, sizeof(res_desc));
                res_desc.resType = cudaResourceTypeArray;
                res_desc.res.array.array = cuda_array_;

                // Specify texture object parameters
                struct cudaTextureDesc tex_desc;
                std::memset(&tex_desc, 0, sizeof(tex_desc));
                tex_desc.addressMode[0] = cudaAddressModeClamp;
                tex_desc.addressMode[1] = cudaAddressModeClamp;
                tex_desc.filterMode = (typeid(T) == typeid(float)) ? cudaFilterModeLinear : cudaFilterModePoint;
                tex_desc.readMode = cudaReadModeElementType;
                tex_desc.normalizedCoords = false;

                // Create texture object
                cuda_assert(cudaCreateTextureObject(&tex_obj_, &res_desc, &tex_desc, NULL), __FILE__, __LINE__);
                cuda_assert(cudaCreateSurfaceObject(&surf_obj_, &res_desc), __FILE__, __LINE__);
            }

            void upload(const T *const data)
            {
                cuda_assert(cudaMemcpy2DToArray(cuda_array_, 0, 0, data, cols_ * sizeof(T), cols_ * sizeof(T), rows_,
                                    cudaMemcpyHostToDevice), __FILE__, __LINE__);
            }

            void download(T *data)
            {
                cuda_assert(cudaMemcpy2DFromArray(data, cols_ * sizeof(T), cuda_array_, 0, 0, cols_ * sizeof(T), rows_,
                                      cudaMemcpyDeviceToHost), __FILE__, __LINE__);
            }

            cudaTextureObject_t &tex_obj()
            {
                return tex_obj_;
            }

            cudaSurfaceObject_t &surf_obj()
            {
                return surf_obj_;
            }

            int rows()
            {
                return rows_;
            }

            int cols()
            {
                return cols_;
            }

        private:

            int rows_, cols_;

            cudaArray *cuda_array_;

            cudaTextureObject_t tex_obj_;

            cudaSurfaceObject_t surf_obj_;
        };
    }
}
#endif //SRC_SURF_TEX_H
