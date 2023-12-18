
import cv2


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def gaussian_blur(img, kernel=3):
    return cv2.GaussianBlur(img, kernel, 0)


def median_blur(img, kernel=3):
    return cv2.medianBlur(img, kernel)
