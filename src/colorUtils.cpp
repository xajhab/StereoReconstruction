// 将此代码添加到disparity.cpp的末尾，或创建单独的colorUtils.cpp文件

namespace ColorUtils {

cv::Mat createColorDisparity(const cv::Mat& disparity, int colormap) {
    if (disparity.empty()) {
        std::cerr << "Error: Input disparity map is empty" << std::endl;
        return cv::Mat();
    }
    
    // Convert disparity map to 8-bit
    cv::Mat disparity8;
    double minVal, maxVal;
    cv::minMaxLoc(disparity, &minVal, &maxVal, nullptr, nullptr, disparity > 0);
    
    if (maxVal > 0) {
        // Normalize only valid disparity
        cv::Mat validMask = disparity > 0;
        disparity.convertTo(disparity8, CV_8U, 255.0/maxVal);
        disparity8.setTo(0, ~validMask); // Set invalid areas to black
    } else {
        disparity8 = cv::Mat::zeros(disparity.size(), CV_8U);
    }
    
    // Apply color mapping
    cv::Mat colorDisparity;
    cv::applyColorMap(disparity8, colorDisparity, colormap);
    
    // Set invalid areas to black
    cv::Mat validMask = disparity > 0;
    colorDisparity.setTo(cv::Scalar(0, 0, 0), ~validMask);
    
    return colorDisparity;
}

cv::Mat blendDisparityWithColor(const cv::Mat& disparity,
                               const cv::Mat& colorImage,
                               double alpha) {
    
    if (disparity.empty() || colorImage.empty()) {
        std::cerr << "Error: Input image is empty" << std::endl;
        return cv::Mat();
    }
    
    if (disparity.size() != colorImage.size()) {
        std::cerr << "Error: Disparity map and color image size do not match" << std::endl;
        return cv::Mat();
    }
    
    // Create color disparity map
    cv::Mat colorDisparity = createColorDisparity(disparity);
    
    if (colorDisparity.empty()) {
        return colorImage.clone();
    }
    
    // Ensure color image is 3 channels
    cv::Mat colorImg3C;
    if (colorImage.channels() == 1) {
        cv::cvtColor(colorImage, colorImg3C, cv::COLOR_GRAY2BGR);
    } else {
        colorImg3C = colorImage.clone();
    }
    
    // Blend images
    cv::Mat blended;
    cv::addWeighted(colorImg3C, 1.0 - alpha, colorDisparity, alpha, 0, blended);
    
    return blended;
}

cv::Mat applyDepthColoring(const cv::Mat& colorImage,
                          const cv::Mat& disparity,
                          float maxDepth) {
    
    if (colorImage.empty() || disparity.empty()) {
        return colorImage.clone();
    }
    
    // Ensure input is 3 channels image
    cv::Mat result;
    if (colorImage.channels() == 1) {
        cv::cvtColor(colorImage, result, cv::COLOR_GRAY2BGR);
    } else {
        result = colorImage.clone();
    }
    
    // Calculate depth weight
    cv::Mat depthWeight;
    cv::Mat validMask = disparity > 0;
    
    // Normalize disparity as depth weight (near objects bright, far objects dark)
    double minVal, maxVal;
    cv::minMaxLoc(disparity, &minVal, &maxVal, nullptr, nullptr, validMask);
    
    if (maxVal > 0) {
        disparity.convertTo(depthWeight, CV_32F, 1.0/maxVal);
        depthWeight.setTo(0, ~validMask);
        
        // Apply depth coloring
        for (int r = 0; r < result.rows; ++r) {
            for (int c = 0; c < result.cols; ++c) {
                if (validMask.at<uchar>(r, c)) {
                    float weight = depthWeight.at<float>(r, c);
                    cv::Vec3b& pixel = result.at<cv::Vec3b>(r, c);
                    
                    // Adjust color intensity based on depth
                    pixel[0] = cv::saturate_cast<uchar>(pixel[0] * weight);
                    pixel[1] = cv::saturate_cast<uchar>(pixel[1] * weight);
                    pixel[2] = cv::saturate_cast<uchar>(pixel[2] * (0.5 + 0.5 * weight)); // Red channel emphasizes near objects
                }
            }
        }
    }
    
    return result;
}

} // namespace ColorUtils

namespace DisparityUtils {

bool postProcessDisparity(cv::Mat& disparity, int filterSize) {
    if (disparity.empty()) {
        return false;
    }
    
    // Median filter to remove noise
    cv::Mat filtered;
    cv::medianBlur(disparity, filtered, filterSize);
    
    // Apply filtered result only in valid areas
    cv::Mat validMask = disparity > 0;
    filtered.copyTo(disparity, validMask);
    
    return true;
}

DisparityQuality analyzeDisparity(const cv::Mat& disparity) {
    DisparityQuality quality;
    
    if (disparity.empty()) {
        return quality;
    }
    
    cv::Mat validMask = disparity > 0;
    quality.totalPixels = disparity.rows * disparity.cols;
    quality.validPixels = cv::countNonZero(validMask);
    quality.validPixelRatio = static_cast<float>(quality.validPixels) / quality.totalPixels;
    
    if (quality.validPixels > 0) {
        cv::Scalar meanScalar = cv::mean(disparity, validMask);
        quality.meanDisparity = static_cast<float>(meanScalar[0]);
        
        double minVal, maxVal;
        cv::minMaxLoc(disparity, &minVal, &maxVal, nullptr, nullptr, validMask);
        quality.disparityRange = static_cast<float>(maxVal - minVal);
    }
    
    return quality;
}

bool saveCompleteResults(const cv::Mat& disparity, 
                       const cv::Mat& leftColor,
                       const cv::Mat& rightColor,
                       const cv::Mat& rectifiedLeftColor,
                       const cv::Mat& rectifiedRightColor,
                       const std::string& basePath) {
    
    try {
        // Save original color images
        if (!leftColor.empty()) {
            cv::imwrite(basePath + "_left_original.png", leftColor);
        }
        if (!rightColor.empty()) {
            cv::imwrite(basePath + "_right_original.png", rightColor);
        }
        
        // Save corrected color images
        if (!rectifiedLeftColor.empty()) {
            cv::imwrite(basePath + "_left_rectified.png", rectifiedLeftColor);
        }
        if (!rectifiedRightColor.empty()) {
            cv::imwrite(basePath + "_right_rectified.png", rectifiedRightColor);
        }
        
        // Save various disparity maps
        if (!disparity.empty()) {
            // Original disparity map
            cv::Mat disparity8;
            disparity.convertTo(disparity8, CV_8U, 255.0/64.0/16.0);
            cv::imwrite(basePath + "_disparity.png", disparity8);
            
            // Color disparity map
            cv::Mat colorDisparity = ColorUtils::createColorDisparity(disparity);
            cv::imwrite(basePath + "_disparity_color.png", colorDisparity);
            
            // Blended image
            if (!rectifiedLeftColor.empty()) {
                cv::Mat blended = ColorUtils::blendDisparityWithColor(
                    disparity, rectifiedLeftColor, 0.3);
                cv::imwrite(basePath + "_disparity_blended.png", blended);
                
                // Depth colored
                cv::Mat depthColored = ColorUtils::applyDepthColoring(
                    rectifiedLeftColor, disparity);
                cv::imwrite(basePath + "_depth_colored.png", depthColored);
            }
        }
        
        std::cout << "[INFO] Complete results saved to: " << basePath << "_*.png" << std::endl;
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "Error saving file: " << e.what() << std::endl;
        return false;
    }
}

} // namespace DisparityUtils