/**
 * @file vio_tools_node.cpp
 * @author Marc-Philip Ecker
 * @date 28.04.20
 */

#include <vio_tools/types/image.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <vio_tools/processing/image_derivatives.h>

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
    const auto t1_x = std::chrono::high_resolution_clock::now();
    vio_tools::Image<int16_t> Ix = vio_tools::sobel_x(I);
    const auto t2_x = std::chrono::high_resolution_clock::now();

    const auto t1_y = std::chrono::high_resolution_clock::now();
    vio_tools::Image<int16_t> Iy = vio_tools::sobel_y(I);
    const auto t2_y = std::chrono::high_resolution_clock::now();

    // Processing time
    const auto dt_x = std::chrono::duration_cast<std::chrono::microseconds>(t2_x - t1_x);
    const auto dt_y = std::chrono::duration_cast<std::chrono::microseconds>(t2_y - t1_y);

    std::cout << "Sobel x: " << dt_x.count() * 1e-6 << "s" << std::endl;
    std::cout << "Sobel y: " << dt_y.count() * 1e-6 << "s" << std::endl;

    // Output
    cv::Mat cv_Ix(Ix.rows(), Ix.cols(), CV_16S), cv_Iy(Ix.rows(), Ix.cols(), CV_16S);

    Ix.download((int16_t *)cv_Ix.data);
    Iy.download((int16_t *)cv_Iy.data);

    cv::convertScaleAbs(cv_Ix, cv_Ix);
    cv::convertScaleAbs(cv_Iy, cv_Iy);

    cv::imshow("Sobel x", cv_Ix);
    cv::imshow("Sobel y", cv_Iy);
    cv::waitKey(0);
}