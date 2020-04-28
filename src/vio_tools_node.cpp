/**
 * @file vio_tools_node.cpp
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */

#include <vio_tools/types/image.h>
#include <vio_tools/types/cuda/surf_tex.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <vio_tools/processing/image_derivatives.h>
#include <vio_tools/processing/cuda/image_derivatives.cuh>

void test_sobel();

int main(int argc, char **argv)
{
    test_sobel();
}

void test_sobel()
{
    // Read OpenCV image
    cv::Mat cv_I = cv::imread("/home/marc/Documents/Datasets/EuRoC/MH_01/mav0/cam0/data/1403636579763555584.png",
                              cv::IMREAD_GRAYSCALE);
    // Convert
    vio_tools::Image<uint8_t> I(cv_I.rows, cv_I.cols, (uint8_t *)cv_I.data);

    // Processing
    const auto t1_hx = std::chrono::high_resolution_clock::now();
    vio_tools::Image<int16_t> hIx = vio_tools::sobel_x(I);
    const auto t2_hx = std::chrono::high_resolution_clock::now();

    const auto t1_hy = std::chrono::high_resolution_clock::now();
    vio_tools::Image<int16_t> hIy = vio_tools::sobel_y(I);
    const auto t2_hy = std::chrono::high_resolution_clock::now();

    vio_tools::Image<int16_t> hIx2(I.rows(), I.cols());
    vio_tools::Image<int16_t> hIy2(I.rows(), I.cols());
    const auto t1_h = std::chrono::high_resolution_clock::now();
    vio_tools::sobel(I, hIx2, hIy2);
    const auto t2_h = std::chrono::high_resolution_clock::now();

    // CUDA processing
    const auto t1_dx = std::chrono::high_resolution_clock::now();
    vio_tools::Image<int16_t> dIx = vio_tools::cuda::sobel_x(I);
    const auto t2_dx = std::chrono::high_resolution_clock::now();

    const auto t1_dy = std::chrono::high_resolution_clock::now();
    vio_tools::Image<int16_t> dIy = vio_tools::cuda::sobel_y(I);
    const auto t2_dy = std::chrono::high_resolution_clock::now();

    vio_tools::Image<int16_t> dIx2(I.rows(), I.cols());
    vio_tools::Image<int16_t> dIy2(I.rows(), I.cols());
    const auto t1_d = std::chrono::high_resolution_clock::now();
    vio_tools::cuda::sobel(I, dIx2, dIy2);
    const auto t2_d = std::chrono::high_resolution_clock::now();

    // Processing time
    const auto dt_hx = std::chrono::duration_cast<std::chrono::microseconds>(t2_hx - t1_hx);
    const auto dt_hy = std::chrono::duration_cast<std::chrono::microseconds>(t2_hy - t1_hy);
    const auto dt_h = std::chrono::duration_cast<std::chrono::microseconds>(t2_h - t1_h);

    const auto dt_dx = std::chrono::duration_cast<std::chrono::microseconds>(t2_dx - t1_dx);
    const auto dt_dy = std::chrono::duration_cast<std::chrono::microseconds>(t2_dy - t1_dy);
    const auto dt_d = std::chrono::duration_cast<std::chrono::microseconds>(t2_d - t1_d);

    std::cout << "Sobel x: " << dt_hx.count() * 1e-6 << "s" << std::endl;
    std::cout << "Sobel y: " << dt_hy.count() * 1e-6 << "s" << std::endl;
    std::cout << "Sobel: " << dt_h.count() * 1e-6 << "s" << std::endl;

    std::cout << "CUDA Sobel x: " << dt_dx.count() * 1e-6 << "s" << std::endl;
    std::cout << "CUDA Sobel y: " << dt_dy.count() * 1e-6 << "s" << std::endl;
    std::cout << "CUDA Sobel: " << dt_d.count() * 1e-6 << "s" << std::endl;

    // Output
    cv::Mat cv_hIx(hIx.rows(), hIx.cols(), CV_16S), cv_hIy(hIx.rows(), hIx.cols(), CV_16S);
    cv::Mat cv_dIx(hIx.rows(), hIx.cols(), CV_16S), cv_dIy(hIx.rows(), hIx.cols(), CV_16S);

    hIx2.download((int16_t *)cv_hIx.data);
    hIy2.download((int16_t *)cv_hIy.data);
    dIx2.download((int16_t *)cv_dIx.data);
    dIy2.download((int16_t *)cv_dIy.data);

    cv::convertScaleAbs(cv_hIx, cv_hIx);
    cv::convertScaleAbs(cv_hIy, cv_hIy);
    cv::convertScaleAbs(cv_dIx, cv_dIx);
    cv::convertScaleAbs(cv_dIy, cv_dIy);

    cv::imshow("Sobel x", cv_hIx);
    cv::imshow("Sobel y", cv_hIy);
    cv::imshow("CUDA Sobel x", cv_dIx);
    cv::imshow("CUDA Sobel y", cv_dIy);
    cv::waitKey(0);
}