import math
import cv2
import indexer
import numpy as np


def main():
    img = cv2.imread('pooltable2.png')
    h, w, _ = img.shape
    img = cv2.resize(img, (1000, round(h * (1000 / w))))
    lower_color, upper_color = detect_cloth_color(img)
    contours = get_cloth_contours(img, lower_color, upper_color)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    epsilon = cv2.arcLength(hull, True) * 0.001
    approx_hull = cv2.approxPolyDP(hull, epsilon, True)
    simplified_lines = simplify_hull_lines(approx_hull)

    print(
        f"Approx hull: {approx_hull} {type(approx_hull)} {approx_hull.shape}")

    print(
        f"Simplified lines: {simplified_lines} {type(simplified_lines)} {simplified_lines.shape}")

    with_contours = cv2.drawContours(
        img.copy(), [largest_contour], -1, (0, 0, 0), 2)

    with_hull = cv2.drawContours(
        img.copy(), [hull], -1, (0, 0, 0), 2)
    for i in range(0, hull.shape[0]):
        cv2.circle(with_hull, hull[i][0], 2, (0, 0, 255), -1)

    with_largest_contour_approx = cv2.drawContours(
        img.copy(), [approx_hull], -1, (0, 0, 0), 2)

    for i in range(0, approx_hull.shape[0]):
        prev_point = approx_hull[(i-1) % approx_hull.shape[0]][0]
        point = approx_hull[i % approx_hull.shape[0]][0]
        next_point = approx_hull[(i+1) % approx_hull.shape[0]][0]

        vec1 = point - prev_point
        vec2 = next_point - point

        angle1 = math.atan2(vec1[1], vec1[0]) * 180.0 / np.pi
        angle2 = math.atan2(vec2[1], vec2[0]) * 180.0 / np.pi
        angle_diff = calc_angle_distance(angle1, angle2)

        mid = (point + next_point) / 2
        mid = [round(mid[0]), round(mid[1])]

        color = (0, 255, 0)
        if (angle_diff < 8):
            color = (255, 0, 0)

        cv2.putText(with_largest_contour_approx,
                    f"{i}: a1: {round(angle1)} a2: {round(angle2)}", mid, fontFace=1, fontScale=0.6, color=(0, 255, 0))
        cv2.putText(with_largest_contour_approx,
                    f"diff: {round(angle_diff)}", [mid[0], mid[1] + 10], fontFace=1, fontScale=0.6, color=(0, 255, 0))
        cv2.circle(with_largest_contour_approx, point, 2, color, -1)

    with_major_lines = cv2.drawContours(
        img.copy(), [simplified_lines], -1, (0, 0, 0), 2)
    for p in simplified_lines:
        point = p[0]
        cv2.circle(with_major_lines, point, 2, (0, 0, 255), -1)

    #cv2.imshow('with_contours', with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imshow('with_hull', with_hull)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('with_largest_contour_approx', with_largest_contour_approx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('major_lines', with_major_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_intersections(hull_lines):
    prev_point = None
    prev_direction = None
    for i in range(hull_lines.shape[0]):
        curr_point = hull_lines[i][0]
        next_point = hull_lines[(i + 1) % hull_lines.shape[0]]
        curr_direction = next_point - curr_point
        len = np.linalg.norm(curr_direction)
        if (len < 100):
            continue

        if (prev_point == None):
            prev_point = curr_point
            prev_direction = curr_direction
            continue

        intersection = [123, 123]

        prev_point = curr_point
        prev_direction = curr_direction


def simplify_hull_lines(hull):
    points = []
    for i in range(hull.shape[0]):
        prev_point = hull[(i-1) % hull.shape[0]][0]
        point = hull[i % hull.shape[0]][0]
        next_point = hull[(i+1) % hull.shape[0]][0]

        vec1 = point - prev_point
        vec2 = next_point - point

        angle1 = math.atan2(vec1[1], vec1[0]) * 180.0 / np.pi
        angle2 = math.atan2(vec2[1], vec2[0]) * 180.0 / np.pi
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)
        total_len = len1 + len2

        angle_diff = calc_angle_distance(angle1, angle2)
        if (angle_diff > 7 and total_len > 70):
            points.append([point])
    return np.array(points)


def calc_angle_distance(angle1, angle2):
    delta = angle2 - angle1
    if (delta > 180):
        delta = delta - 360
    if (delta < -180):
        delta = delta + 360
    return delta


def detect_cloth_color(img, hue_width=55, sat_width=45, val_width=210):
    """
    Find the most common HSV values in the image.
    In a well lit image, this will be the cloth
    """
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    # cv2.imshow('HSV', hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    h, w, _ = hsv.shape
    mask = np.zeros((h, w, 1), dtype=np.uint8)
    cv2.rectangle(mask, (round(0.2*w), round(0.1*h)),
                  (round(0.8*w), round(0.9*h)), (255, 255, 255), -1)

    # cv2.imshow('Masked HSL', cv2.bitwise_and(hsv, hsv, mask=mask))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    h_max = indexer.get_index_of_max(hist)[0]

    hist = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
    s_max = indexer.get_index_of_max(hist)[0]

    hist = cv2.calcHist([hsv], [2], mask, [256], [0, 256])
    v_max = indexer.get_index_of_max(hist)[0]

    hsv_max = (h_max, s_max, v_max)
    print(f"Most common HSV: {hsv_max}")

    # Create a blank 300x300 black image
    color_image = np.zeros((300, 300, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    color_image[:] = hsv_max
    color_image = cv2.cvtColor(color_image, cv2.COLOR_HSV2BGR)

    # cv2.imshow('Main color', color_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # define range of blue color in HSV
    lower_color = np.array(
        [h_max-hue_width, s_max-sat_width, v_max-val_width])
    upper_color = np.array(
        [h_max+hue_width, s_max+sat_width, v_max+val_width])

    return lower_color, upper_color


def get_cloth_contours(img, lower_color, upper_color, filter_radius=5):
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    """
    Returns the contours generated from the given color range
    """

    # Threshold the HSV image to get only cloth colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # use a median filter to get rid of speckle noise
    median = cv2.medianBlur(mask, filter_radius)
    """
    cv2.imshow('Median', median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=5)
    """
    cv2.imshow('Dilated', dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    shrunk = cv2.erode(dilated, kernel, iterations=5)

    cv2.imshow('Shrunk', shrunk)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(
        shrunk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


main()
