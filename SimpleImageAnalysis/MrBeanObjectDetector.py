# Mr Bean Object Detector: This program is a proof-of-concept object detector that detects a broom, wheels and ...
# road line in an image of Mr Bean (see MrBeanObjectDetectorOriginal.png).
# 0 Set-Up
import cv2 as cv
import numpy as np

image_bgr = cv.imread("MrBeanObjectDetectorOriginal.png")
image_hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

# 1 Line Search
# 1.1 Line Pre-Processing
image_filtered_rgb = cv.inRange(image_rgb, np.array([200, 200, 200]), np.array([255, 255, 255]))
image_filtered = cv.cvtColor(image_filtered_rgb, cv.COLOR_RGB2BGR)
image_gray = cv.cvtColor(image_filtered, cv.COLOR_BGR2GRAY)
image_canny = cv.Canny(image_gray, 50, 150)
lines = cv.HoughLinesP(image_canny, rho = 1, theta = np.pi / 180, 
                       threshold = 50, minLineLength = 50, maxLineGap = 20)
# 1.2 Line Detection and Back Projection
if lines is not None:
    for i in lines:
        x1, y1, x2, y2 = i[0]
        slope = (y2 - y1) / (x2 - x1 + 0.01)
        if (abs(slope) > 4) or abs(slope) < 0.2:
            cv.line(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 5)

# 2 Circle Search
# 2.1 Line Pre-Processing
mask = cv.inRange(image_hsv, np.array([0, 0, 0]), np.array([200, 50, 100]))
image_filtered_hsv = cv.bitwise_and(image_hsv, image_bgr, mask = mask)
image_filtered = cv.cvtColor(image_filtered_hsv, cv.COLOR_HSV2BGR)
image_gray = cv.cvtColor(image_filtered, cv.COLOR_BGR2GRAY)
image_canny = cv.Canny(image_gray, 50, 150)
circles = cv.HoughCircles(image_canny, cv.HOUGH_GRADIENT, dp = 1, minDist = 20, 
                          param1 = 50, param2 = 30, minRadius = 20, maxRadius = 50)
# 2.2 Circle Detection and Back Projection
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        centre = (circle[0], circle[1])
        radius = circle[2]
        cv.circle(image_bgr, centre, radius, (0, 255, 0), 5)
        cv.circle(image_bgr, centre, 5, (0, 255, 0), -1)

# 3 Solution Display
cv.imshow("Detected Features", image_bgr)
cv.waitKey(0)
cv.destroyAllWindows()
