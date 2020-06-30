import cv2
import numpy as np


def findCorners(img):
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ = cv2.GaussianBlur(img_, (3, 3), 0)
    img_ = cv2.bitwise_not(img_)
    AdaptiveThreshold = cv2.adaptiveThreshold(img_, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    line_length = int(0.25 * min(img.shape[0], img.shape[1]))
    lines = cv2.HoughLinesP(AdaptiveThreshold, 1, np.pi / 180, 60, minLineLength=line_length, maxLineGap=10)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()
    scale = 200

    horizontalSize = int(horizontal.shape[1] / scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalsize = int(vertical.shape[1] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    corners_img = cv2.bitwise_and(horizontal, vertical)

    # cv2.imshow("AdaptiveThreshold", AdaptiveThreshold)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("line_detect_possible", img)
    # cv2.imshow("horizontal", horizontal)
    # cv2.imshow("verticalsize", vertical)
    # cv2.imshow("corners_img", corners_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(corners_img, connectivity=8)
    return centroids, lines


def crossCounterFilter(img, centroids, v_more=10, h_more=5, v_diff=2, h_diff=10):
    centroids = sorted(centroids, key=lambda x: x[0])
    vertical_counters = []
    counter = 1
    for i in range(1, len(centroids)):
        if abs(centroids[i][0] - centroids[i - 1][0]) <= v_diff:
            counter += 1
        else:
            vertical_counters += [counter] * counter
            counter = 1
    vertical_counters += [counter] * counter
    vertical_dict = dict()
    for i in range(len(centroids)):
        vertical_dict[tuple(centroids[i])] = vertical_counters[i]

    centroids = sorted(centroids, key=lambda x: x[1])
    horizontal_counters = []
    counter = 1
    for i in range(1, len(centroids)):
        if abs(centroids[i][1] - centroids[i - 1][1]) <= h_diff:
            counter += 1
        else:
            horizontal_counters += [counter] * counter
            counter = 1
    horizontal_counters += [counter] * counter
    horizontal_dict = dict()
    for i in range(len(centroids)):
        horizontal_dict[tuple(centroids[i])] = horizontal_counters[i]

    new_centroids = list()
    for centroid in centroids:
        if vertical_dict[tuple(centroid)] >= v_more and horizontal_dict[tuple(centroid)] >= h_more:
            # print(vertical_dict[tuple(centroid)], horizontal_dict[tuple(centroid)])
            # cv2.circle(img, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 255), 2)
            # cv2.imshow('img', img)
            # cv2.waitKey()
            new_centroids.append(centroid)
    return new_centroids


def nmsToLineDistanceFilter(points, lines):
    spacing = line_spacing(lines)
    points_distance = corners_min_distance_to_lines(points, lines)
    points = none_max_suppress(points, points_distance, spacing // 2)
    return points


def line_spacing(lines):
    lines = lines[:, 0, :]
    lines = lines[lines[:, 3] - lines[:, 1] == 0]
    lines_x = sorted(lines[:, 3])

    clusters_lines_x = list()
    l = 0
    for r in range(1, len(lines_x)):
        if lines_x[r] - lines_x[r - 1] > 5:
            clusters_lines_x.append(int(np.mean(lines_x[l:r])))
            l = r
    spacings = list()
    for i in range(1, len(clusters_lines_x)):
        spacings.append(clusters_lines_x[i] - clusters_lines_x[i - 1])
    counts = np.bincount(spacings)
    mode = np.argmax(counts)
    return mode


def none_max_suppress(points, points_distance, radius, distance_threshold=8):
    arg = points_distance <= distance_threshold
    points = points[arg]
    points_distance = points_distance[arg]

    arg = points_distance.argsort()
    points = points[arg]

    i = 0
    while i < len(points):
        j = i + 1
        print(len(points))
        while j < len(points):
            norm = np.linalg.norm(points[i] - points[j])
            if norm < radius:
                points = np.delete(points, j, axis=0)
            else:
                j += 1
        i += 1
    return points


def corners_min_distance_to_lines(points: np.ndarray, lines: np.ndarray):
    points_x = points[:, [0]]
    points_y = points[:, [1]]

    lines = lines[:, 0, :].T
    lines_x1 = lines[[0], :]
    lines_y1 = lines[[1], :]
    lines_x2 = lines[[2], :]
    lines_y2 = lines[[3], :]

    lines_k = (lines_y2 - lines_y1) / (lines_x2 - lines_x1)
    lines_k[np.isinf(lines_k)] = 1000000000000
    distance = (lines_k * (points_x - lines_x1) - (points_y - lines_y1)) / (np.sqrt(lines_k * lines_k + 1))
    points_distance = np.min(np.abs(distance), axis=1)
    return points_distance


if __name__ == '__main__':
    input_Path = '2.jpg'
    # input_Path = 'decolor.png'
    input_Path = 'ren.png'
    img = cv2.imread(input_Path)
    corners, lines = findCorners(img)

    line_spacing(lines)

    for corner in corners:
        cv2.circle(img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), 3)

        # put_str = str(int(corner[0]))
        # # put_str = str(int(corner[1]))
        # img = cv2.putText(img, put_str, (int(corner[0]), int(corner[1])),
        #                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, )
        # cv2.imwrite("corner_test0.png", img)

    corners = nmsToLineDistanceFilter(corners, lines)
    for i in range(len(corners)):
        cv2.circle(img, (int(corners[i][0]), int(corners[i][1])), 6, (0, 255, 255), 3)

    corners = crossCounterFilter(img, corners)
    for i in range(len(corners)):
        cv2.circle(img, (int(corners[i][0]), int(corners[i][1])), 4, (255, 0, 0), 3)
        # cv2.circle(img, (int(corners[i][0]), int(corners[i][1])), 2, (0, 255, 0), 3)
        # put_str = str(int(vertical_counters[i]))
        # img = cv2.putText(img, put_str, (int(corners[i][0]), int(corners[i][1])),
        #                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, )

    cv2.imwrite("corner_test1.png", img)

    arr = np.array([1, 2, 3])
    ccc = arr[arr < 2]
    print(ccc)
