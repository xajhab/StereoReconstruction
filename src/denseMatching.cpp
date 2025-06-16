#include "denseMatching.h"
#include <iostream>

DenseMatcher::DenseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                           int numDisparities, int blockSize)
    : K_(K.clone()), distCoeffs_(distCoeffs.clone()),
      numDisparities_(numDisparities), blockSize_(blockSize) {
}

bool DenseMatcher::rectifyImages(const cv::Mat& imgL, const cv::Mat& imgR,
                                 const cv::Mat& R, const cv::Mat& t,
                                 cv::Mat& rectL, cv::Mat& rectR) {

    // Stereo rectification
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K_, distCoeffs_, K_, distCoeffs_,
                      imgL.size(), R, t, R1, R2, P1, P2, Q);

    Q_ = Q.clone(); // Store the Q matrix

    // Generate rectification maps
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(K_, distCoeffs_, R1, P1,
                                imgL.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K_, distCoeffs_, R2, P2,
                                imgR.size(), CV_32FC1, mapRx, mapRy);

    // Apply rectification
    cv::remap(imgL, rectL, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(imgR, rectR, mapRx, mapRy, cv::INTER_LINEAR);

    return true;
}

bool DenseMatcher::computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                                        cv::Mat& disparity) {

    // Create stereo matcher
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(numDisparities_, blockSize_);

    // Set parameters
    stereo->setPreFilterCap(31);
    stereo->setBlockSize(blockSize_);
    stereo->setMinDisparity(0);
    stereo->setNumDisparities(numDisparities_);
    stereo->setTextureThreshold(10);
    stereo->setUniquenessRatio(15);
    stereo->setSpeckleWindowSize(100);
    stereo->setSpeckleRange(32);
    stereo->setDisp12MaxDiff(1);

    // Compute disparity map
    stereo->compute(rectL, rectR, disparity);

    return !disparity.empty();
}

const cv::Mat& DenseMatcher::getQMatrix() const {
    return Q_;
} 