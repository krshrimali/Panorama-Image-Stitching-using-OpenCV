import cv2
import numpy as np
import argparse

def resize(src, dst, width=None, height=None,
        inter=cv2.INTER_AREA):
    dim = None
    (h, w) = src.shape[:2] # take height and width
    
    # if nothing of width and height is given, return image
    if width is None and height is None:
        return src 
    if width is None:
        # ratio of new height to original height
        ratio = height / (float)
        # change width according to the ratio
        dim = (int(w * ratio), height)
    else:
        ratio = weight/(float)
        dim = (weight, int(h * ratio))
    
    result = cv2.resize(src, dim, interpolation=inter)

    return result
    
def stitcher(images, drawKP, reprojThresh = 4.0, ratio = 0.75):
    (imageB, imageA) = images #unwrap images from left to right
    
    # resize both images to 400 width size
    width = 400 
    imageA = resize(imageA, width)
    imageB = resize(imageB, width)

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

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

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

if imageA is None:
    print("None")
images = [imageA, imageB]

(imageA, imageB, result) = stitcher(images, 1)
cv2.imshow("imgA", imageA)
cv2.imshow("imgB", imageB)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
