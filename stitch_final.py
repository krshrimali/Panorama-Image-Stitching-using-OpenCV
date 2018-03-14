import cv2

def resize(src, dst, ratio):
    dst = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)
    return dst

def stitcher(images, drawKP):
    (imageA, imageB) = images #unwrap images from left to right
    
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # detect features and key points
    (kpA, featuresA) = detectFeaturesKeys(grayA)
    (kpB, featuresB) = detectFeaturesKeys(grayB)
    
    if(drawKP == 1):
        imageA = cv2.drawKeypoints(grayA, kpA, imageA)
        imageB = cv2.drawKeypoints(grayB, kpB, imageB)
    
    result = (imageA, imageB)

    return result

def detectFeaturesKeys(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(image, None) 

    return (kp, sift)
def show(result):
    (imageA, imageB) = result

    cv2.imshow("imageA", imageA)
    cv2.imshow("imageB", imageB)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

imageA = cv2.imread("../Task-2/images/bryce_left_01.png", 1)
imageB = cv2.imread("../Task-2/images/bryce_right_01.png", 1)

images = (imageA, imageB)

result = stitcher(images, 1)
show(result)
