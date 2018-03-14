import cv2
import numpy as np
def resize(src, dst, ratio):
    dst = cv2.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)
    return dst

def stitcher(images, drawKP, reprojThresh = 4.0, ratio = 0.75):
    (imageB, imageA) = images #unwrap images from left to right
    

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

    return warped_image

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


imageA = cv2.imread("../Task-2/images/first.jpg", 1)
imageB = cv2.imread("../Task-2/images/second.jpg", 1)

if imageA is None:
    print("None")
images = [imageB, imageA]

result = stitcher(images, 1)
cv2.imshow("imgA", imageA)
cv2.imshow("imgB", imageB)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
