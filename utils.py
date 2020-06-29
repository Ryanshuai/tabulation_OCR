import numpy as np


def decolor(img):
    img = (img / 255).astype(np.float32)
    ret, img = cv2.threshold(img, 0.4, 1., cv2.THRESH_BINARY)
    img = (1 - (1 - img[:, :, 0]) * (1 - img[:, :, 1]) * (1 - img[:, :, 2]))
    img = (img * 255).astype(np.uint8)
    return img


def remove_isolated_pixels(image):
    connectivity = 100

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image


def remove_debris(img, threshold=1000):
    contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < threshold:
            cv2.drawContours(img, [contours[i]], -1, (84, 1, 68), thickness=-1)  # 原始图片背景BGR值(84,1,68)
            continue
    cv2.imshow('clean_img', img)
    cv2.waitKey()
    return img


if __name__ == '__main__':
    import cv2

    img = cv2.imread('ren.png')

    decolored = decolor(img)
    cv2.imshow('removed', decolored)
    cv2.waitKey()

    clean_img = remove_isolated_pixels(decolored)
    cv2.imshow('clean_img', clean_img)
    cv2.waitKey()

    cv2.imwrite('decolor.png', clean_img)
