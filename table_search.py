import cv2
import numpy as np


def find_table_corners(img):
    corners, lines, h_lines, v_lines = findCorners(img)
    clusters_rows_y, clusters_columns_x = row_column_clustering(lines)
    spacing, min_line_number = row_num_spacing(clusters_rows_y)

    h_num, h_centers = horizontal_line_cluster(h_lines, spacing // 2)
    v_num, v_centers = vertical_line_cluster(v_lines, spacing // 2)

    corners = nmsToLineDistanceFilter(corners, lines, spacing)
    corners = crossCounterFilter(corners)

    arrange_standard_rectangle_centers(corners)

    return corners


def findCorners(img):
    # img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ = cv2.GaussianBlur(img, (3, 3), 0)
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
    horizontal_lines = cv2.HoughLinesP(horizontal, 1, np.pi / 180, 60, minLineLength=line_length, maxLineGap=10)

    verticalsize = int(vertical.shape[1] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    vertical_lines = cv2.HoughLinesP(vertical, 1, np.pi / 180, 60, minLineLength=line_length, maxLineGap=10)

    corners_img = cv2.bitwise_and(horizontal, vertical)

    # cv2.imshow("AdaptiveThreshold", AdaptiveThreshold)
    # img_line, img_horizontal, img_vertical = img.copy(), img.copy(), img.copy()
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img_line, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # for line in horizontal_lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img_horizontal, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # for line in vertical_lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img_vertical, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("line_detect_possible", img_line)
    # cv2.imshow("horizontal", horizontal)
    # cv2.imshow("horizontal_line_detect_possible", img_horizontal)
    # cv2.imshow("verticalsize", vertical)
    # cv2.imshow("vertical_line_detect_possible", img_vertical)
    # cv2.imshow("corners_img", corners_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(corners_img, connectivity=8)
    return centroids, lines, horizontal_lines, vertical_lines


def nmsToLineDistanceFilter(points, lines, spacing):
    points_distance = corners_min_distance_to_lines(points, lines)
    points = none_max_suppress(points, points_distance, spacing // 2)
    return points


def horizontal_line_cluster(lines, distance_threshold=5):
    t12s = np.concatenate((lines[:, 0, [1]], lines[:, 0, [3]]), axis=1)
    return scan_cluster(t12s, distance_threshold)


def vertical_line_cluster(lines, distance_threshold=5):
    t12s = np.concatenate((lines[:, 0, [0]], lines[:, 0, [2]]), axis=1)
    return scan_cluster(t12s, distance_threshold)


def scan_cluster(t12s, gap_fill):
    t12s = set([(y1, y2) for y1, y2 in t12s])
    start_points = [(y1, -1) for y1, y2 in t12s]
    end_points = [(y2, 1) for y1, y2 in t12s]
    t_s = sorted(start_points + end_points)

    cluster_centers = []
    t_in_one_cluster = []
    t0, t_counter, cluster_counter = 0, 0, 0
    for t, flag in t_s:
        if t_counter <= 0 and t - t0 > gap_fill:
            cluster_counter += 1
            if t_in_one_cluster:
                cluster_centers.append(sum(t_in_one_cluster) // len(t_in_one_cluster))
                t_in_one_cluster = []
        t_in_one_cluster.append(t)
        t_counter -= flag
        t0 = t
    cluster_centers.append(sum(t_in_one_cluster) // len(t_in_one_cluster))
    return cluster_counter, cluster_centers


def row_column_clustering(lines):
    lines = lines[:, 0, :]
    rows = lines[lines[:, 3] - lines[:, 1] == 0]
    rows_y = sorted(rows[:, 3])
    columns = lines[lines[:, 2] - lines[:, 0] == 0]
    columns_x = sorted(columns[:, 2])

    clusters_rows_y = list()
    l = 0
    for r in range(1, len(rows_y)):
        if rows_y[r] - rows_y[r - 1] > 5:
            clusters_rows_y.append(int(np.mean(rows_y[l:r])))
            l = r

    clusters_columns_x = list()
    l = 0
    for r in range(1, len(columns_x)):
        if columns_x[r] - columns_x[r - 1] > 5:
            clusters_columns_x.append(int(np.mean(columns_x[l:r])))
            l = r

    return clusters_rows_y, clusters_columns_x


def row_num_spacing(clusters_rows_y):
    rows_spacings = list()
    for i in range(1, len(clusters_rows_y)):
        rows_spacings.append(clusters_rows_y[i] - clusters_rows_y[i - 1])
    counts = np.bincount(rows_spacings)
    rows_spacing = np.argmax(counts)
    min_row_number = max(counts)
    return rows_spacing, min_row_number


def none_max_suppress(points, points_distance, radius, distance_threshold=8):
    arg = points_distance <= distance_threshold
    points = points[arg]
    points_distance = points_distance[arg]

    arg = points_distance.argsort()
    points = points[arg]

    i = 0
    while i < len(points):
        j = i + 1
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
    distance = np.partition(np.abs(distance), range(2))
    points_distance = np.mean(distance[:, :2], axis=1)
    return points_distance


def crossCounterFilter(centroids, v_more=10, h_more=5, v_diff=2, h_diff=10):
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
            new_centroids.append(centroid)
    return np.array(new_centroids)


def arrange_standard_rectangle_centers(centroids, ):
    pass


def twoClosetVerticalHorizontalFilter(points):  # TODO
    points_x = points[:, [0]]
    points_y = points[:, [1]]
    distance_2 = np.power((points_x - points_x.T), 2) + np.power((points_y - points_y.T), 2)
    arg = np.argpartition(distance_2, range(3))[:, 1:3]
    diff = abs(points[arg] - points[:, np.newaxis, :])
    min_k = np.min(diff, axis=-1) / np.max(diff, axis=-1)
    two_closest_VH_count_bigger_than_1 = np.sum(min_k < np.tan(np.pi / 18), axis=-1) >= 1
    points = points[two_closest_VH_count_bigger_than_1]
    return points


if __name__ == '__main__':
    input_path = 'input_images/2.jpg'
    # input_path = 'decolor.png'
    # input_path = 'ren.png'
    print("input_path : ", input_path)
    img = cv2.imread(input_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, lines, h_lines, v_lines = findCorners(gray_img)
    print("find lines : ", len(corners))

    clusters_rows_y, clusters_columns_x = row_column_clustering(lines)
    spacing, min_line_number = row_num_spacing(clusters_rows_y)
    print("line_spacing : ", spacing)
    print("min_line_number : ", min_line_number)

    for corner in corners:
        cv2.circle(img, (int(corner[0]), int(corner[1])), 8, (0, 0, 255), 3)

        # put_str = str(int(corner[0]))
        # # put_str = str(int(corner[1]))
        # img = cv2.putText(img, put_str, (int(corner[0]), int(corner[1])),
        #                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, )
        # cv2.imwrite("corner_test0.png", img)

    corners = nmsToLineDistanceFilter(corners, lines, 14)
    print("nmsToLineDistanceFilter result lines: ", len(corners))
    for i in range(len(corners)):
        cv2.circle(img, (int(corners[i][0]), int(corners[i][1])), 6, (0, 255, 255), 3)

    # corners = crossCounterFilter(img, corners, v_more=min_line_number)
    corners = crossCounterFilter(corners)
    print("crossCounterFilter result lines: ", len(corners))
    for i in range(len(corners)):
        cv2.circle(img, (int(corners[i][0]), int(corners[i][1])), 4, (255, 0, 0), 3)
        # cv2.circle(img, (int(corners[i][0]), int(corners[i][1])), 2, (0, 255, 0), 3)
        # put_str = str(int(vertical_counters[i]))
        # img = cv2.putText(img, put_str, (int(corners[i][0]), int(corners[i][1])),
        #                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, )

    # corners = twoClosetVerticalHorizontalFilter(corners)
    # print("twoClosetVerticalHorizontalFilter result lines: ", len(corners))
    # for i in range(len(corners)):
    #     cv2.circle(img, (int(corners[i][0]), int(corners[i][1])), 2, (0, 255, 0), 3)

    cv2.imwrite("corner_test.png", img)
