#Build a panorama stitching program that combines two images into one panoramic image.import cv2
import numpy as np
import cv2

# Load images
img1 = cv2.imread("img1.png")  # left image
img2 = cv2.imread("img2.png")  # right image

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Use BFMatcher
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Find Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image1 to image2 plane
    height, width = img2.shape[:2]
    result = cv2.warpPerspective(img1, H, (width + img1.shape[1], height))

    # Place image2 in result
    result[0:height, 0:width] = img2

    # Show result
    cv2.imshow("Panorama", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Not enough matches found!")