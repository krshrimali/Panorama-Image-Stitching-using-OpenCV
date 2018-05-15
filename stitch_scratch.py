'''
Panorama using OpenCV (Python)

Usage: python stitch_final.py --first <image path> --second <image path>

Author: Kushashwa Ravi Shrimali
'''
# importing necessary libraries
import cv2
import numpy as np
import argparse
from scipy import ndimage
from image_align import *

# utility function - resize image to defined height and width
def resize(src, width=None, height=None, inter=cv2.INTER_AREA):
    '''
    Utility Function: Resizes Image to new width and height

    Mention new width and height, else None expected.
    Default interpolation - INTER_AREA
    '''

    print("Resizing... ")
    # set dimension to None, to return if no dimension specified
    dim = None
    (h, w) = src.shape[:2] # take height and width

    # handle None values in width and height
    if width is None and height is None:
        return src 

    # to resize, aspect ratio has to be kept in check
    # for no distortion of the shape of the imag3
    if width is None:
        # calculate aspect ratio changed according to new height
        ratio = height / float(h)
        # change width according to the ratio
        dim = (int(w * ratio), height)
    else:
        # calculate aspect ratio changed according to new width
        ratio = width/float(w)
        dim = (width, int(h * ratio))
    # apply resizing using calculated dimensions 
    result = cv2.resize(src, dim, interpolation=inter)
    return result

def detectFeaturesKeys(image):
    '''
    Detects features and key points.
    Return : (keypoints, features)
    For OpenCV 3.x only (to-do : opencv 2.x)

    Usage: detectFeaturesKeys(<source image>)
    '''
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    kps = np.float32([kp_.pt for kp_ in kps])
    return (kps, features)

def draw_keyPoints_BFMatcher(imageA, kpA, imageB, kpB, featuresA, featuresB):
    # reference : OpenCV Tutorial for Drawing Key Points (Brute Force method)
    good = []
    print("Showing matches")
    
    bf = cv2.BFMatcher()
    matches_new = bf.knnMatch(featuresA, featuresB, k = 2)

    for m, n in matches_new:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    
    # passing None for output image
	# details
    # https://stackoverflow.com/questions/31631352/typeerror-required-argument-outimg-pos-6-not-found 
    # cv2.drawMatchesKnn 
    img_drawn = cv2.drawMatchesKnn(imageA, kpA, imageB, kpB, good, None, flags=2)
    cv2.imshow("Key Points", img_drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    '''
    Match Key Points
    Usage: matchKeyPoints(imageA, imageB, keypoints_imageA, keypoints_imageB, features_imageA,
    features_imageB, ratio, reprojThresh)

    Returns: None if no match found
    (matches, H, status) if matches found.
    '''
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    matches = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    
    # writing function to debug newspaper image panorama issue
    # draw_keyPoints_BFMatcher(imageA, kpA, imageB, kpB, featuresA, featuresB)
    

    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return (matches, H, status)
    return None


def stitcher(images, reprojThresh = 4.0, ratio = 0.75):
    '''
    Stitcher function : stitches the two images to form the panorama 
    Usage: stitcher(<image list: imageA, imageB>, reprojThresh = 4.0, ratio = 0.75)

    returns list of three images: [imageA, imageB, panorama_image]
    '''
    # unwrap images from right to left
    # imageB = images[0]
    # imageA = images[1]

    (imageB, imageA) = images 
    show(imageB)
    show(imageA)

    # detect features and key points
    (kpsA, featuresA) = detectFeaturesKeys(imageA)
    (kpsB, featuresB) = detectFeaturesKeys(imageB)

    # match features between the two images 
    matched_features = matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    
    if(matched_features is None):
        print("No features matched.")
        return None

    # unwrap homography matrix (H), matches and status 
    (matches, H, status) = matched_features

    # warping transformation 
    warped_image = cv2.warpPerspective(imageA, H,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    warped_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    # wrap all the images to a list and return
    result = warped_image
    return result

def show(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate_and_crop(result_):
    # select ROI from image
    # reference: https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
    r = cv2.selectROI(result_, False)
    result_ = result_[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    
    rotate_ = input("Do you want to rotate the image? (yes/no) ")
    if(rotate_.lower() == "yes"):
        degrees = int(input("Degree by which you want to rotate the image: "))
        
        rows, cols, channels = result_.shape
        M          = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
        dst        = cv2.warpAffine(result_, M, (cols,rows))
        
        result_ = ndimage.rotate(result_, degrees)
        show(result_)
    
    return result_

if __name__ == "__main__":
    # construct argument parse
    arg = argparse.ArgumentParser() 

    arg.add_argument("-dest", "--destination", required=True, help = "Destination Name")
    
    args = vars(arg.parse_args())

    # count_of_images = int(input("Number of images to stitch: "))
    initial = "images/test2/"
    images_list = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg"]
    images_list = [initial + x for x in images_list]
    print(images_list)
    count = 0

    while True:
        if(count == 0):
            imageA = cv2.imread(images_list[count])
            imageB = cv2.imread(images_list[count + 1])
            imageA       = resize(imageA, 400)
            imageB       = resize(imageB, 400)
            print(imageA.shape, imageB.shape)
            count += 2
        else:
            imageB       = cv2.imread(images_list[count])
            imageB       = resize(imageB, 400)
            print(imageA.shape, imageB.shape)
            count += 1
        images           = [imageA, imageB]
        imageA           = stitcher(images)
        # imageA = resize(imageA, 400)
        show(imageA)
        print(imageA.shape, imageB.shape)
        
        # imageA = rotate_and_crop(imageA)
        
        if(count == len(images_list)):
            break
    # imageA = rotate_and_crop(imageA)
    show(imageA)
    cv2.imwrite(args["destination"], imageA)

    '''
    images = [imageB, imageA]
    
    (imageA, imageB, result_A_B) = stitcher(images, 1)
    
    show(result_A_B)

    # result_A_B = resize(result_A_B, 400)

    images = [imageC, result_A_B]
    (imageC, result_A_B, result_AB_C) = stitcher(images, 1)
    show(result_AB_C)

    images = [imageD, result_AB_C]
    (imageD, result_AB_C, result_AB_CD) = stitcher(images, 1)
    
    # result_ = resize(result_, 400)
    
    # select ROI from image
    # reference: https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
    r = cv2.selectROI(result_, False)
    result_ = result_[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    
    rotate_ = input("Do you want to rotate the image? (yes/no) ")
    if(rotate_.lower() == "yes"):
        degrees = int(input("Degree by which you want to rotate the image: "))
        
        rows, cols, channels = result_.shape
        M          = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
        dst        = cv2.warpAffine(result_, M, (cols,rows))
        
        result_ = ndimage.rotate(result_, degrees)
        show(result_)
     
    # write the image to the destination
    cv2.imwrite(args["destination"], result_)  
    '''
