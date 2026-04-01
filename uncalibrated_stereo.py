import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# USER INPUTS
# =========================================================
LEFT_IMAGE_PATH = "left.jpeg"
RIGHT_IMAGE_PATH = "right.jpeg"

baseline_m = 0.05      # 5 cm
ground_truth_m = 0.217  # 21.7 cm

# Approximate intrinsic matrix for 1600x1200 image
f_px = 1250.0
cx = 800.0
cy = 600.0

K = np.array([
    [f_px, 0, cx],
    [0, f_px, cy],
    [0, 0, 1]
], dtype=np.float64)

# =========================================================
# LOAD IMAGES
# =========================================================
img1 = cv2.imread(LEFT_IMAGE_PATH)
img2 = cv2.imread(RIGHT_IMAGE_PATH)

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not load left.jpeg or right.jpeg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print("Left image shape :", img1.shape)
print("Right image shape:", img2.shape)

# =========================================================
# DETECT AND MATCH FEATURES
# =========================================================
orb = cv2.ORB_create(3000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise ValueError("Could not compute descriptors. Try different images.")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches_knn = bf.knnMatch(des1, des2, k=2)

good_matches = []
for pair in matches_knn:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print(f"Total keypoints in left image  : {len(kp1)}")
print(f"Total keypoints in right image : {len(kp2)}")
print(f"Good matches after ratio test  : {len(good_matches)}")

if len(good_matches) < 8:
    raise ValueError("Not enough good matches to compute Fundamental Matrix")

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# =========================================================
# FUNDAMENTAL MATRIX
# =========================================================
F, mask = cv2.findFundamentalMat(
    pts1, pts2, cv2.FM_RANSAC, 1.5, 0.99
)

if F is None or F.shape != (3, 3):
    raise ValueError("Fundamental Matrix could not be estimated")

inlier_pts1 = pts1[mask.ravel() == 1]
inlier_pts2 = pts2[mask.ravel() == 1]

print("\nNumber of inlier matches:", len(inlier_pts1))
print("\nFundamental Matrix (F):")
print(F)

# =========================================================
# ESSENTIAL MATRIX
# =========================================================
E = K.T @ F @ K

print("\nEssential Matrix (E):")
print(E)

# =========================================================
# ROTATION MATRIX
# =========================================================
retval, R, t, pose_mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K)

print("\nRecovered pose inliers:", retval)
print("\nRotation Matrix (R):")
print(R)

print("\nTranslation direction (t):")
print(t)

# =========================================================
# DRAW MATCHES FOR REPORT
# =========================================================
match_vis = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches[:80], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("feature_matches.jpg", match_vis)

# =========================================================
# HELPER FUNCTION: SHOW RESIZED IMAGE AND MAP CLICKS BACK
# =========================================================
clicked_points = []

def show_resized(window_name, img, max_width=1000):
    h, w = img.shape[:2]
    if w <= max_width:
        resized = img.copy()
        scale = 1.0
    else:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
    return resized, scale

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked displayed point: ({x}, {y})")

def get_point_from_user(img, window_name):
    global clicked_points
    clicked_points = []

    display_img, scale = show_resized(window_name, img, max_width=1000)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = display_img.copy()
        for p in clicked_points:
            cv2.circle(temp, p, 6, (0, 0, 255), -1)
        cv2.imshow(window_name, temp)

        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) >= 1:
            break
        if key == 27:
            break

    cv2.destroyWindow(window_name)

    if len(clicked_points) < 1:
        raise ValueError(f"No point selected in {window_name}")

    disp_x, disp_y = clicked_points[0]

    # Convert displayed image coordinates back to original image coordinates
    orig_x = int(disp_x / scale)
    orig_y = int(disp_y / scale)

    return (orig_x, orig_y), (disp_x, disp_y), scale

# =========================================================
# MANUAL POINT SELECTION FOR DISPARITY
# =========================================================
print("\nNow select the SAME point on the bottle in both images.")
print("Recommended: center of cap, or a clear point on the label.")

left_pt, left_disp_pt, left_scale = get_point_from_user(
    img1, "Left Image - Click one point on the object"
)

right_pt, right_disp_pt, right_scale = get_point_from_user(
    img2, "Right Image - Click the SAME point on the object"
)

print("\nSelected point in original left image :", left_pt)
print("Selected point in original right image:", right_pt)

# =========================================================
# DISPARITY AND DEPTH
# =========================================================
x_left = left_pt[0]
x_right = right_pt[0]

disparity = abs(x_left - x_right)

if disparity == 0:
    raise ValueError("Disparity is zero, so depth cannot be computed")

estimated_depth_m = (f_px * baseline_m) / disparity
absolute_error_m = abs(estimated_depth_m - ground_truth_m)
percent_error = (absolute_error_m / ground_truth_m) * 100

print(f"\nDisparity (pixels): {disparity:.2f}")
print(f"Estimated Distance from camera setup: {estimated_depth_m:.4f} m")
print(f"Ground Truth Distance: {ground_truth_m:.4f} m")
print(f"Absolute Error: {absolute_error_m:.4f} m")
print(f"Percentage Error: {percent_error:.2f}%")

# =========================================================
# ANNOTATE FINAL IMAGE
# =========================================================
annotated = img1.copy()

# Draw selected point
cv2.circle(annotated, left_pt, 8, (0, 0, 255), -1)

# Bounding box around approximate sanitizer region
# Adjust if needed, but this should work reasonably for your image
# Better bounding box around sanitizer
x1, y1, x2, y2 = 600, 330, 910, 910
cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)

cv2.putText(
    annotated,
    "Selected Object: Hand Sanitizer",
    (40, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 0, 0),
    2
)

cv2.putText(
    annotated,
    f"Estimated Distance: {estimated_depth_m * 100:.2f} cm",
    (40, 115),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 255),
    2
)

cv2.putText(
    annotated,
    f"Ground Truth Distance: {ground_truth_m * 100:.2f} cm",
    (40, 170),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 128, 255),
    2
)

cv2.putText(
    annotated,
    f"Disparity: {disparity:.2f} px",
    (40, 225),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (128, 0, 255),
    2
)

cv2.imwrite("annotated_result.jpg", annotated)

# =========================================================
# OPTIONAL: EPIPOLAR LINES VISUALIZATION
# =========================================================
def draw_epilines(img1, img2, lines, pts1, pts2):
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    r, c = img1_copy.shape[:2]

    for r_line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        a, b, c_line = r_line[0]
        if abs(b) > 1e-6:
            x0, y0 = 0, int(-c_line / b)
            x1_line, y1_line = img1_copy.shape[1], int(-(c_line + a * img1_copy.shape[1]) / b)
            cv2.line(img1_copy, (x0, y0), (x1_line, y1_line), color, 1)
        cv2.circle(img1_copy, tuple(np.int32(pt1)), 5, color, -1)
        cv2.circle(img2_copy, tuple(np.int32(pt2)), 5, color, -1)

    return img1_copy, img2_copy

try:
    sample_n = min(10, len(inlier_pts1))
    sample_pts1 = inlier_pts1[:sample_n]
    sample_pts2 = inlier_pts2[:sample_n]

    lines1 = cv2.computeCorrespondEpilines(sample_pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 1, 3)

    epi_left, epi_right = draw_epilines(img1, img2, lines1, sample_pts1, sample_pts2)
    cv2.imwrite("epilines_left.jpg", epi_left)
    cv2.imwrite("epilines_right.jpg", epi_right)
except Exception as e:
    print("\nCould not generate epipolar line images:", e)

# =========================================================
# SHOW RESULTS
# =========================================================
plt.figure(figsize=(16, 10))

plt.subplot(2, 1, 1)
plt.title("Feature Matches")
plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 1, 2)
plt.title("Annotated Final Result")
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nSaved files:")
print("1. feature_matches.jpg")
print("2. annotated_result.jpg")
print("3. epilines_left.jpg (if generated)")
print("4. epilines_right.jpg (if generated)")