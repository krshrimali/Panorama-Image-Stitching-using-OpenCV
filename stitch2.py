import cv2

image_left = cv2.imread(input("First image: "), 1)
image_right = cv2.imread(input("Second image: "), 1)

# resize the images
cv2.resize(image_left, image_left, Size(), 0.5, 0.5, interpolation)
cv2.resize(image_right, image_right, Size(), 0.5, 0.5, interpolation)

# checking commit 
cv2.is_v3() # checks if opencv version is 3.3+ or not

cv2.imshow("image_left", image_left)
cv2.imshow("image_right", image_right)

cv2.waitKey(0)
cv2.DestroyAllWindows()
