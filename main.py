import os
import cv2


from table_search import find_table_corners


def main():
    input_dir = "input_images"
    output_dir = "output_excel"

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        img = cv2.imread(image_path)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = find_table_corners(gray_img)

        for i in range(len(corners)):
            cv2.circle(img, (int(corners[i][0]), int(corners[i][1])), 4, (255, 0, 0), 3)

        print("find lines : ", len(corners))
        cv2.imwrite("corner_main_test.png", img)




if __name__ == '__main__':
    main()