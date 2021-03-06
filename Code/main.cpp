/*
 * main.cpp
 *
 *  Created on: 28 May 2016
 *      Author: Minh Ngo @ 3DUniversum
 */
#include <iostream>
#include <boost/format.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/marching_cubes.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/impl/texture_mapping.hpp>
#include <pcl/features/normal_3d_omp.h>

#include <eigen3/Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>

#include "Frame3D/Frame3D.h"

#include <typeinfo>

pcl::PointCloud<pcl::PointXYZ>::Ptr mat2IntegralPointCloud(const cv::Mat& depth_mat, const float focal_length, const float max_depth) {
    // This function converts a depth image to a point cloud
    assert(depth_mat.type() == CV_16U);
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    const int half_width = depth_mat.cols / 2;
    const int half_height = depth_mat.rows / 2;
    const float inv_focal_length = 1.0 / focal_length;
    point_cloud->points.reserve(depth_mat.rows * depth_mat.cols);
    for (int y = 0; y < depth_mat.rows; y++) {
        for (int x = 0; x < depth_mat.cols; x++) {
            float z = depth_mat.at<ushort>(cv:: Point(x, y)) * 0.001;
            if (z < max_depth && z > 0) {
                point_cloud->points.emplace_back(static_cast<float>(x - half_width)  * z * inv_focal_length,
                                                 static_cast<float>(y - half_height) * z * inv_focal_length,
                                                 z);
            } else {
                point_cloud->points.emplace_back(x, y, NAN);
            }
        }
    }
    point_cloud->width = depth_mat.cols;
    point_cloud->height = depth_mat.rows;
    return point_cloud;
}


pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    // This function computes normals given a point cloud
    // !! Please note that you should remove NaN values from the pointcloud after computing the surface normals.
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>); // Output datasets
    pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);
    ne.compute(*cloud_normals);
    pcl::copyPointCloud(*cloud, *cloud_normals);
    return cloud_normals;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const Eigen::Matrix4f& transform) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
    return transformed_cloud;
}

template<class T>
typename pcl::PointCloud<T>::Ptr transformPointCloudNormals(typename pcl::PointCloud<T>::Ptr cloud, const Eigen::Matrix4f& transform) {
    typename pcl::PointCloud<T>::Ptr transformed_cloud(new typename pcl::PointCloud<T>());
    pcl::transformPointCloudWithNormals(*cloud, *transformed_cloud, transform);
    return transformed_cloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mergingPointClouds(Frame3D frames[]) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    for (int i = 0; i < 8; i++) {
        std::cout << boost::format("Merging frame %d") % i << std::endl;

        Frame3D frame = frames[i];
        cv::Mat depthImage = frame.depth_image_;
        double focalLength = frame.focal_length_;
        const Eigen::Matrix4f cameraPose = frame.getEigenTransform();
		float threshold = 1;

        // Merge the i-th frame using predicted camera pose to the global point cloud
		
		// Point cloud <- depthToPointCloud(depth image, focal length)
		pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>());
		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
		pointCloud = mat2IntegralPointCloud(depthImage, focalLength, threshold);
		//pcl::copyPointCloud(*pointCloud, *pointCloudRGB);
		
		// Point cloud with normals <- computeNormals(point cloud)
	    pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointNormal>);
		cloudNormals = computeNormals(pointCloud);
		
		// Remove NaNs from normals
	    pcl::PointCloud<pcl::PointNormal>::Ptr filteredCloudNormals(new pcl::PointCloud<pcl::PointNormal>);
	    std::vector<int> indices;
		pcl::removeNaNFromPointCloud(*cloudNormals, *filteredCloudNormals, indices);
				
		// Pass the point cloud with normals
		pcl::PointCloud<pcl::PointNormal>::Ptr newPointCloud(new pcl::PointCloud<pcl::PointNormal>);
	    pcl::transformPointCloudWithNormals(*filteredCloudNormals, *newPointCloud, cameraPose);
		
		// Add it to the total cloud
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr copiedCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
		pcl::copyPointCloud(*newPointCloud, *copiedCloud);
		*modelCloud += *copiedCloud;
	}
    return modelCloud;
}


pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mergingPointCloudsWithTexture(Frame3D frames[]) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	for (int i = 0; i < 8; i++) {
        std::cout << boost::format("Merging frame %d") % i << std::endl;

        Frame3D frame = frames[i];
        cv::Mat depthImage = frame.depth_image_;
        double focalLength = frame.focal_length_;
        const Eigen::Matrix4f cameraPose = frame.getEigenTransform();

        // TODO(Student): The same as mergingPointClouds but now with texturing. ~ 50 lines.
    }
    return modelCloud;
}

// Different methods of constructing mesh
enum CreateMeshMethod { PoissonSurfaceReconstruction = 0, MarchingCubes = 1};

// Create mesh from point cloud using one of above methods
pcl::PolygonMesh createMesh(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointCloud, CreateMeshMethod method) {
    std::cout << "Creating meshes" << std::endl;
	
    // The variable for the constructed mesh
    pcl::PolygonMesh triangles;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*pointCloud, *cloudNormals);
	
    switch (method) {
        case PoissonSurfaceReconstruction:
		{
            // Poisson Surface Reconstruction
			pcl::Poisson<pcl::PointNormal> poisson;
			poisson.setDepth(8);
			poisson.setSolverDivide(8);
			poisson.setIsoDivide(8);
			poisson.setPointWeight(4.0f);
			poisson.setInputCloud(cloudNormals);
			poisson.reconstruct(triangles);
			break;
		}
        case MarchingCubes:
            // Marching Cubes Surface Reconstruction
			pcl::MarchingCubesHoppe<pcl::PointNormal> marchingCubes;
			marchingCubes.setIsoLevel(0.0f);
			marchingCubes.setGridResolution(50, 50, 50);
			marchingCubes.setPercentageExtendGrid(0.0f);
		    marchingCubes.setInputCloud(cloudNormals);
		    marchingCubes.reconstruct(triangles);
            break;
    }
    return triangles;
}


int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cout << "./final [3DFRAMES PATH] [RECONSTRUCTION MODE] [TEXTURE_MODE]" << std::endl;

        return 0;
    }

    const CreateMeshMethod reconMode = static_cast<CreateMeshMethod>(std::stoi(argv[2]));

    // Loading 3D frames
    Frame3D frames[8];
    for (int i = 0; i < 8; ++i) {
        frames[i].load(boost::str(boost::format("%s/%05d.3df") % argv[1] % i));
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr texturedCloud;
    pcl::PolygonMesh triangles;

    if (argv[3][0] == 't') {
        // SECTION 4: Coloring 3D Model
        // Create one point cloud by merging all frames with texture using
        // the rgb images from the frames
        texturedCloud = mergingPointCloudsWithTexture(frames);

        // Create a mesh from the textured cloud using a reconstruction method,
        // Poisson Surface or Marching Cubes
        triangles = createMesh(texturedCloud, reconMode);
    } else {
        // SECTION 3: 3D Meshing & Watertighting

        // Create one point cloud by merging all frames with texture using
        // the rgb images from the frames
        texturedCloud = mergingPointClouds(frames);

        // Create a mesh from the textured cloud using a reconstruction method,
        // Poisson Surface or Marching Cubes
        triangles = createMesh(texturedCloud, reconMode);
    }

    // Sample code for visualization.

    // Show viewer
    std::cout << "Finished texturing" << std::endl;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    // Add colored point cloud to viewer, because it does not support colored meshes
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(texturedCloud);
    viewer->addPointCloud<pcl::PointXYZRGBNormal>(texturedCloud, rgb, "cloud");

    // Add mesh
    viewer->setBackgroundColor(1, 1, 1);
    viewer->addPolygonMesh(triangles, "meshes", 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Keep viewer open
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }


    return 0;
}
