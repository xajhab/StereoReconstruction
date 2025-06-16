#ifndef DEPTH_H
#define DEPTH_H

#include <opencv2/opencv.hpp>

namespace Depth {
bool computeDepthMap(const cv::Mat& disparity,
                     const cv::Mat& Q,
                     cv::Mat& depthMap,
                     const std::string& outputPath);
} // namespace Depth

#endif // DEPTH_H 