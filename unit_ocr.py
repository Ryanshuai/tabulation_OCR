import pytesseract
import cv2

if __name__ == '__main__':
    img = cv2.imread('uestc.png')
    print(pytesseract.image_to_string(img, lang='chi_sim'))
