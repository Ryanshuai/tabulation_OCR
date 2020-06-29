import cv2


def findCorners(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.bitwise_not(img)
    AdaptiveThreshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()
    scale = 20

    horizontalSize = int(horizontal.shape[1] / scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalsize = int(vertical.shape[1] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    corners_img = cv2.bitwise_and(horizontal, vertical)
    # contours, hierarchy = cv2.findContours(horizontal + vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("horizontal", horizontal)
    # cv2.imshow("verticalsize", vertical)
    # cv2.imshow("mask", mask)
    # cv2.imshow("Net_img", Net_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    output = cv2.connectedComponents(corners_img, connectivity=8, ltype=cv2.CV_32S)

    return img, corners_img




if __name__ == '__main__':
    # input_Path = '2.png'
    input_Path = 'ren.png'
    src_img = cv2.imread(input_Path)
    src_img, Net_img = findCorners(src_img)

    cv2.imshow("src_img", src_img)
    cv2.imshow("Net_img", Net_img)
    cv2.waitKey()

