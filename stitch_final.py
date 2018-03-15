import cv2
import numpy as np
import argparse

def resize(src, width=None, height=None, inter=cv2.INTER_AREA):
    print("Resizing... ")
    dim = None
    (h, w) = src.shape[:2] # take height and width
    if width is None and height is None:
        return src 
    if width is None:
        # ratio of new height to original height
        ratio = height / float(h)
        # change width according to the ratio
        dim = (int(w * ratio), height)
    else:
        ratio = width/float(w)
        dim = (width, int(h * ratio))
    
    result = cv2.resize(src, dim, interpolation=inter)
    return result
    
def stitcher(images, drawKP, reprojThresh = 4.0, ratio = 0.75):
    (imageB, imageA) = images #unwrap images from left to right
    print(imageA.shape) 

    # detect features and key points
    (kpA, featuresA) = detectFeaturesKeys(imageA)
    (kpB, featuresB) = detectFeaturesKeys(imageB)
    
    
    matched_features = matchKeyPoints(kpA, kpB, featuresA,
            featuresB, ratio, reprojThresh)

    if(matched_features is None):
        print("No features detected.")
        return None

    (matches, H, status) = matched_features

    # warping transformation 
    warped_image = cv2.warpPerspective(imageA, H,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    warped_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    
    result = [imageA, imageB, warped_image]

    return result

def detectFeaturesKeys(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kp, features) = descriptor.detectAndCompute(image, None)
    
    kps = np.float32([kp_.pt for kp_ in kp])
    return (kps, features)

def matchKeyPoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    matches = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return (matches, H, status)
    return None

# construct argument parse
arg = argparse.ArgumentParser() 

# parse arguments
arg.add_argument("-f", "--first", required=True,
        help = "first image path")
arg.add_argument("-s", "--second", required=True,
        help = "second image path")

args = vars(arg.parse_args())

# read images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

# resizing images to width 400
imageA = resize(imageA, 400)
imageB = resize(imageB, 400)

# wrap both images in a list
images = [imageA, imageB]

(imageA, imageB, result) = stitcher(images, 1)

cv2.imshow("imgA", imageA)
cv2.imshow("imgB", imageB)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
