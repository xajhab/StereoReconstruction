#pragma once
#include <opencv2/opencv.hpp>

void run_stereo_reconstruction(const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& K, const cv::Mat& distCoeffs); 