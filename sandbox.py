import cv2 as cv

img = cv.imread("resources/image_1.png")
cv.imshow('sample image', img, )
cv.waitKey(0)
cv.destroyAllWindows()
