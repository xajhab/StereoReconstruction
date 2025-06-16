#include "meshReconstruction.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

namespace MeshReconstruction {

// Global parameters
static ReconstructionParams g_params;

void setReconstructionParams(const ReconstructionParams& params) {
    g_params = params;
    std::cout << "[INFO] Mesh reconstruction parameters updated" << std::endl;
}

bool reconstructAndSaveMesh(const cv::Mat& depthMap, 
                           const cv::Mat& colorImage, 
                           const std::string& outputPath) {
    
    std::cout << "[INFO] Start mesh reconstruction..." << std::endl;
    std::cout << "[INFO] Depth map size: " << depthMap.size() << ", type: " << depthMap.type() << std::endl;
    std::cout << "[INFO] Color image size: " << colorImage.size() << ", type: " << colorImage.type() << std::endl;
    std::cout << "[INFO] Output path: " << outputPath << std::endl;
    
    if (depthMap.empty()) {
        std::cerr << "Error: Depth map is empty" << std::endl;
        return false;
    }
    
    // Default camera intrinsics (should be passed from outside, using default here)
    cv::Mat K = (cv::Mat_<double>(3,3) << 700, 0, 320, 0, 700, 240, 0, 0, 1);
    
    // Generate point cloud
    std::vector<Point3D> pointCloud = generatePointCloud(depthMap, colorImage, K);
    
    if (pointCloud.empty()) {
        std::cerr << "Error: Failed to generate point cloud" << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Generated point cloud, number of points: " << pointCloud.size() << std::endl;
    
    // Save as PLY format
    std::string plyPath = outputPath;
    size_t pos = plyPath.find_last_of('.');
    if (pos != std::string::npos) {
        plyPath = plyPath.substr(0, pos) + ".ply";
    } else {
        plyPath += ".ply";
    }
    
    bool success = savePointCloudPLY(pointCloud, plyPath);
    
    if (success) {
        std::cout << "[SUCCESS] Mesh reconstruction completed, saved to: " << plyPath << std::endl;
        
        // Also save a simplified point cloud file
        std::string simplifiedPath = plyPath;
        pos = simplifiedPath.find_last_of('.');
        if (pos != std::string::npos) {
            simplifiedPath = simplifiedPath.substr(0, pos) + "_simplified.ply";
        }
        
        // Simplify point cloud (sample one every 5 points)
        std::vector<Point3D> simplifiedCloud;
        for (size_t i = 0; i < pointCloud.size(); i += 5) {
            simplifiedCloud.push_back(pointCloud[i]);
        }
        
        if (savePointCloudPLY(simplifiedCloud, simplifiedPath)) {
            std::cout << "[INFO] Simplified version saved to: " << simplifiedPath << std::endl;
        }
    } else {
        std::cerr << "[ERROR] Mesh reconstruction failed" << std::endl;
    }
    
    return success;
}

bool reconstructFromDisparity(const cv::Mat& disparityMap,
                             const cv::Mat& Q,
                             const cv::Mat& colorImage,
                             const std::string& outputPath) {
    
    if (disparityMap.empty() || Q.empty()) {
        std::cerr << "Error: Disparity map or Q matrix is empty" << std::endl;
        return false;
    }
    
    // Convert disparity to depth
    cv::Mat xyz;
    cv::reprojectImageTo3D(disparityMap, xyz, Q, true);
    
    // Extract depth information
    std::vector<cv::Mat> xyzChannels;
    cv::split(xyz, xyzChannels);
    cv::Mat depthMap = xyzChannels[2];
    
    // Call depth map reconstruction method
    return reconstructAndSaveMesh(depthMap, colorImage, outputPath);
}

std::vector<Point3D> generatePointCloud(const cv::Mat& depthMap, 
                                       const cv::Mat& colorImage,
                                       const cv::Mat& K) {
    
    std::vector<Point3D> points;
    
    // Get camera intrinsics
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    
    // Check depth map type and convert
    cv::Mat processedDepth;
    if (depthMap.type() == CV_16U) {
        depthMap.convertTo(processedDepth, CV_32F, 1.0/1000.0); // Assume unit is millimeter
    } else if (depthMap.type() == CV_32F) {
        processedDepth = depthMap.clone();
    } else {
        std::cerr << "Warning: Unsupported depth map type: " << depthMap.type() << std::endl;
        depthMap.convertTo(processedDepth, CV_32F);
    }
    
    bool hasColor = !colorImage.empty() && (colorImage.size() == depthMap.size());
    
    int step = 1; // Sampling step, can be adjusted to control point cloud density
    
    for (int y = 0; y < processedDepth.rows; y += step) {
        for (int x = 0; x < processedDepth.cols; x += step) {
            float depth = processedDepth.at<float>(y, x);
            
            // Filter invalid depth values
            if (depth <= 0 || depth > g_params.depthThreshold || std::isnan(depth) || std::isinf(depth)) {
                continue;
            }
            
            // Back-project to 3D space
            float worldX = (x - cx) * depth / fx;
            float worldY = (y - cy) * depth / fy;
            float worldZ = depth;
            
            Point3D point(worldX, worldY, worldZ);
            
            // Add color information
            if (hasColor && g_params.useColor) {
                if (colorImage.channels() == 3) {
                    cv::Vec3b color = colorImage.at<cv::Vec3b>(y, x);
                    point.b = color[0]; // BGR format
                    point.g = color[1];
                    point.r = color[2];
                } else if (colorImage.channels() == 1) {
                    uint8_t gray = colorImage.at<uint8_t>(y, x);
                    point.r = point.g = point.b = gray;
                }
            } else {
                // Color by depth
                uint8_t colorValue = static_cast<uint8_t>(255 * (1.0f - std::min(depth / 5000.0f, 1.0f)));
                point.r = colorValue;
                point.g = colorValue;
                point.b = 255 - colorValue;
            }
            
            points.push_back(point);
        }
    }
    
    std::cout << "[INFO] Generated " << points.size() << " 3D points from " << processedDepth.rows << "x" << processedDepth.cols << " depth map" << std::endl;
    
    return points;
}

bool savePointCloudPLY(const std::vector<Point3D>& points, 
                      const std::string& filename) {
    
    if (points.empty()) {
        std::cerr << "Error: Point cloud is empty, cannot save" << std::endl;
        return false;
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return false;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    // Write point data
    for (const auto& point : points) {
        file << point.x << " " << point.y << " " << point.z << " "
             << static_cast<int>(point.r) << " " 
             << static_cast<int>(point.g) << " " 
             << static_cast<int>(point.b) << "\n";
    }
    
    file.close();
    
    std::cout << "[INFO] Successfully saved " << points.size() << " points to " << filename << std::endl;
    return true;
}

} // namespace MeshReconstruction