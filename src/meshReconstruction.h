#ifndef MESH_RECONSTRUCTION_H
#define MESH_RECONSTRUCTION_H

#include <opencv2/opencv.hpp>
#include <string>

namespace MeshReconstruction {
    
    /**
     * Reconstruct and save mesh from depth map and color image
     * @param depthMap Depth map (CV_32F or CV_16U)
     * @param colorImage Color image (optional, for texture)
     * @param outputPath Output file path (.ply or .obj)
     * @return Success or not
     */
    bool reconstructAndSaveMesh(const cv::Mat& depthMap, 
                               const cv::Mat& colorImage, 
                               const std::string& outputPath);
    
    /**
     * Reconstruct and save mesh from disparity map
     * @param disparityMap Disparity map
     * @param Q Reprojection matrix
     * @param colorImage Color image
     * @param outputPath Output file path
     * @return Success or not
     */
    bool reconstructFromDisparity(const cv::Mat& disparityMap,
                                 const cv::Mat& Q,
                                 const cv::Mat& colorImage,
                                 const std::string& outputPath);
    
    /**
     * Mesh reconstruction parameters
     */
    struct ReconstructionParams {
        float depthThreshold = 10000.0f;  // Depth threshold
        float voxelSize = 1.0f;           // Voxel size
        bool useColor = true;             // Use color or not
        bool smoothMesh = true;           // Smooth mesh or not
        int decimationTarget = 100000;    // Target triangle count for mesh decimation
    };
    
    /**
     * Set reconstruction parameters
     */
    void setReconstructionParams(const ReconstructionParams& params);
    
    /**
     * Point cloud data structure
     */
    struct Point3D {
        float x, y, z;
        uint8_t r, g, b;
        
        Point3D() : x(0), y(0), z(0), r(0), g(0), b(0) {}
        Point3D(float x_, float y_, float z_, uint8_t r_ = 0, uint8_t g_ = 0, uint8_t b_ = 0)
            : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
    };
    
    /**
     * Generate point cloud from depth map
     */
    std::vector<Point3D> generatePointCloud(const cv::Mat& depthMap, 
                                           const cv::Mat& colorImage,
                                           const cv::Mat& K);
    
    /**
     * Save point cloud as PLY format
     */
    bool savePointCloudPLY(const std::vector<Point3D>& points, 
                          const std::string& filename);
    
} // namespace MeshReconstruction

#endif // MESH_RECONSTRUCTION_H