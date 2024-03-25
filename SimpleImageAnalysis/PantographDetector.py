# 0 Set-Up
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

x_target = 1000  # x_target: x value of intersection point (from top left corner).
y_target = 1000  # y_target: y value of intersection point (from top left corner).
target_line_slope = 0  # target_line_slope: the slope of the line associated with the intersection point.
alpha = 1  # alpha: hyperparameter for smoothing.
n_steps = 0  # n_steps: number of steps of detected intersection points.
n_frame = 0  # n_frame: the frame of the video being analysed.
n_frames = []  # n_frames: the frames at which intersection points are detected.
x_trajectory = []  # x_trajectory: the x trajectory of the intersection point.
y_trajectory = []  # y_trajectory: the y trajectory of the intersection point.

# 1 Loading and Pre-Processing
video = cv.VideoCapture("C:/Users/sjp20/OneDrive/Documents/2024/Semester 1/ELEC4630/A1 Resources/Panto2024.mp4")
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    frame = frame[100:frame.shape[0] - 100, 100:frame.shape[1] - 100]
    frame_size = frame.shape
    frame_area = frame_size[0] * frame_size[1]
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_blurred = cv.GaussianBlur(frame_gray, (3, 3), 3)
    frame_canny = cv.Canny(frame_blurred, 40, 130)

# 2 Probabilistic Hough Line Transform
    lines = cv.HoughLinesP(frame_canny, rho = 1, theta = np.pi/180, threshold = 50, minLineLength = 150, maxLineGap = 30)

# 2.1 Line Filtering (Slope)
    vertical_lines = []
    horizontal_lines = []
    if lines is not None:
        for i in lines:
            x1, y1, x2, y2 = i[0]
            slope = (y2 - y1) / (x2 - x1 + 0.01)
            if slope > 4:
                vertical_lines.append(i)
            if abs(slope) < 0.1:
                horizontal_lines.append(i)   

# 2.2 Intersection Points Calculation and Further Line Filtering
    intersections = []
    vertical_intersect_lines = []
    horizontal_intersect_lines = []
    for i, line_i in enumerate(vertical_lines):
        for j, line_j in enumerate(horizontal_lines):
            x1_i, y1_i, x2_i, y2_i = line_i[0]
            x1_j, y1_j, x2_j, y2_j = line_j[0]
            slope_i = (y2_i - y1_i) / (x2_i - x1_i + 0.1) 
            slope_j = (y2_j - y1_j) / (x2_j - x1_j + 0.1) 

            x_intersect = (y1_j - y1_i + slope_i * x1_i - slope_j * x1_j) / (slope_i - slope_j)
            y_intersect = slope_i * (x_intersect - x1_i) + y1_i

            if ((min(x1_i, x2_i) - 0.1 <= x_intersect <= max(x1_i, x2_i) + 0.1 and
                min(x1_j, x2_j) - 0.1 <= x_intersect <= max(x1_j, x2_j) + 0.1 and
                min(y1_i, y2_i) - 0.1 <= y_intersect <= max(y1_i, y2_i) + 0.1 and
                min(y1_j, y2_j) - 0.1 <= y_intersect <= max(y1_j, y2_j) + 0.1)):

                vertical_intersect_lines.append(line_i)
                cv.line(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
                cv.line(frame, (x1_j, y1_j), (x2_j, y2_j), (0, 255, 0), 2)
                horizontal_intersect_lines.append(line_j)
                intersections.append((x_intersect, y_intersect))

# 3 Intersection Point Filtering
    if vertical_intersect_lines == [] or horizontal_intersect_lines == [] or intersections == []:
        pass
    else:
        for i in vertical_intersect_lines:
            x1, y1, x2, y2 = i[0]
            target_line_slope = (y2 - y1) / (x2 - x1 + 0.01) 
       
        x_min, y_min = intersections[0]
        for intersection in intersections:
            x, y = intersection
            if y < y_min:
                if target_line_slope > 0:
                    if x < x_min:
                        x_min, y_min = intersection
                        cv.circle(frame, (int(x_target), int(y_target)), 10, (255, 0, 0), -1)
                elif x > x_min:
                    x_min, y_min = intersection
                    cv.circle(frame, (int(x_target), int(y_target)), 10, (255, 0, 0), -1)

# 4 Intersection Point Update
        if abs(y_min - y_target) < (alpha ** 4) * 1000 and abs(x_min - x_target) < (alpha ** 2) * 200:
            y_target = alpha * y_min + (1 - alpha) * y_target
            x_target = alpha * x_min + (1 - alpha) * x_target
            y_target = alpha * y_min + (1 - alpha) * y_target
            
            if n_steps < 20:
                alpha *= 0.95
            else:
                n_frames.append(n_frame)
                x_trajectory.append(x_target + 100)
                y_trajectory.append(y_target + 100)
            
            n_steps += 1

# 5 Solution Display
# 5.1 Image Solution
    cv.circle(frame, (int(x_target), int(y_target)), 5, (0, 0, 255), -1)
    cv.imshow('Frame', frame)
    cv.waitKey(10)
    n_frame += 1

video.release()

# 5.2 Plot Solution
plt.clf()
plt.figure(figsize = (12, 6))

plt.subplot(1, 2, 1)
plt.plot(n_frames, x_trajectory)
plt.xlabel("Time (Frame)")
plt.ylabel("X Position")
plt.title("X Trajectory")

plt.subplot(1, 2, 2)
plt.plot(n_frames, y_trajectory, color = "red")
plt.xlabel("Time (Frame)")
plt.ylabel("Y Position (Pixels)")
plt.title("Y Trajectory")

plt.show()
