// CPP program to Stitch
// input images (panorama) using OpenCV 
// Reference : https://docs.opencv.org/3.3.0/d8/d19/tutorial_stitcher.html
// Credits : https://www.geeksforgeeks.org/stitching-input-images-panorama-using-opencv-c/
#include <iostream>
#include <fstream>
 
// Include header files from OpenCV directory
// required to stitch images.
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
 
using namespace std;
using namespace cv;
 
// Define mode for stitching as panoroma 
// (One out of many functions of Stitcher)
Stitcher::Mode mode = Stitcher::PANORAMA;
 
// Array for pictures
vector<Mat> imgs;
 
int main(int argc, char* argv[])
{
    // Get all the images that need to be 
    // stitched as arguments from command line 
    for (int i = 1; i < argc; ++i)
    {
            // Read the ith argument or image 
            // and push into the image array
            Mat img = imread(argv[i]);
            if (img.empty())
            {
                // Exit if image is not present
                cout << "Can't read image '" << argv[i] << "'\n";
                return -1;
            }
            imgs.push_back(img);
    }
     
    // Define object to store the stitched image
    Mat pano;
     
    // Create a Stitcher class object with mode panoroma
    Ptr<Stitcher> stitcher = Stitcher::create(mode, false);
     
    // Command to stitch all the images present in the image array
    Stitcher::Status status = stitcher->stitch(imgs, pano);
 
    if (status != Stitcher::OK)
    {
        // Check if images could not be stiched
        // status is OK if images are stiched successfully
        cout << "Can't stitch images\n";
        return -1;
    }
     
    // Store a new image stiched from the given 
    //set of images as "result.jpg"
    imwrite("result_newspaper.jpg", pano);
     
    // Show the result
    imshow("Result", pano);
     
    waitKey(0);
    return 0;
}