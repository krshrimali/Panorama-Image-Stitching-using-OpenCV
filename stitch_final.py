'''
Panorama using OpenCV (Python)

Usage: python stitch_final.py --first <image path> --second <image path>

Author: Kushashwa Ravi Shrimali
'''
# importing necessary libraries
import cv2
import numpy as np
import argparse

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
    (kp, features) = descriptor.detectAndCompute(image, None)

    kps = np.float32([kp_.pt for kp_ in kp])
    return (kp, kps, features)

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
    cv2.drawMatchesKnn 
    img_drawn = cv2.drawMatchesKnn(imageA, kpA, imageB, kpB, good, None, flags=2)
    cv2.imshow("Key Points", img_drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def matchKeyPoints(kpA, kpB, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    '''
    Match Key Points
    Usage: matchKeyPoints(keypoints_imageA, keypoints_imageB, features_imageA,
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
    draw_keyPoints_BFMatcher(imageA, kpA, imageB, kpB, featuresA, featuresB)
    

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
    (imageB, imageA) = images 

    # detect features and key points
    (kpA, kpsA, featuresA) = detectFeaturesKeys(imageA)
    (kpB, kpsB, featuresB) = detectFeaturesKeys(imageB)

    # match features between the two images 
    matched_features = matchKeyPoints(kpA, kpB, kpsA, kpsB, featuresA,
            featuresB, ratio, reprojThresh)
    
    # draw_keypoints(imageB, kpB, imageA, kpA, featuresA, featuresB)
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
    result = [imageA, imageB, warped_image]
    return result

def show(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # construct argument parse
    arg = argparse.ArgumentParser() 

    # parse arguments
    arg.add_argument("-f", "--first", required=True,
            help = "first image path")
    arg.add_argument("-s", "--second", required=True,
            help = "second image path")
    arg.add_argument("-t", "--third", required=False, help = "third image path")
    arg.add_argument("-fo", "--fourth", required=False, help = "fourth image path")
    arg.add_argument("-dest", "--destination", required=True, help = "Destination Name")

    args = vars(arg.parse_args())

    # read images
    imageA = cv2.imread(args["first"])
    imageB = cv2.imread(args["second"])

    imageC = cv2.imread(args["third"])
    imageD = cv2.imread(args["fourth"])

    imageA = resize(imageA, 400)
    imageB = resize(imageB, 400)
    imageC = resize(imageC, 400)
    imageD = resize(imageD, 400)

    # wrap both images in a list
    images = [imageB, imageA]
    
    (imageA, imageB, result_A_B) = stitcher(images, 1)
    
    show(result_A_B)

    # result_A_B = resize(result_A_B, 400)

    images = [imageC, result_A_B]
    (imageC, result_A_B, result_AB_C) = stitcher(images, 1)
    show(result_AB_C)

    images = [imageD, result_AB_C]
    (imageD, result_AB_C, result_AB_CD) = stitcher(images, 1)

    # result_AB_CD = resize(result_AB_CD, 400)
    # images = [imageC, result_A_B]
    # (imageC, result_A_B, result_AB_C) = stitcher(images, 1)

    cv2.imwrite(args["destination"], result_AB_CD)
    cv2.imshow('result', result_AB_CD)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
