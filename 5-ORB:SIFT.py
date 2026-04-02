import cv2

img1 = cv2.imread('sample.jpeg', 0)
img2 = cv2.imread('gray_image.jpg', 0)

# ORB FEATURE MATCHING

orb = cv2.ORB_create(nfeatures=1000)

kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des1_orb, des2_orb)

matches_orb = sorted(matches_orb, key=lambda x: x.distance)

orb_result = cv2.drawMatches(
    img1, kp1_orb,
    img2, kp2_orb,
    matches_orb[:50], None
)

#  SIFT FEATURE MATCHING

sift = cv2.SIFT_create()

kp1_sift, des1_sift = sift.detectAndCompute(img1, None)
kp2_sift, des2_sift = sift.detectAndCompute(img2, None)

bf_sift = cv2.BFMatcher(cv2.NORM_L2)

matches_sift = bf_sift.knnMatch(des1_sift, des2_sift, k=2)

# Lowe's Ratio Test
good_matches = []
for m, n in matches_sift:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

sift_result = cv2.drawMatches(
    img1, kp1_sift,
    img2, kp2_sift,
    good_matches, None
)

cv2.imshow("ORB Feature Matching", orb_result)
cv2.imshow("SIFT Feature Matching", sift_result)

cv2.waitKey(0)
cv2.destroyAllWindows()