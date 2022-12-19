import cv2 as cv
import numpy

width = 480
height = 640


def prepare_image(image):
    # convert the image to grayscale format
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # apply binary thresholding
    ret, thresh = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
    return thresh


def get_contour(image):
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    biggest = detect_biggest_contour(contours)
    return biggest


def detect_biggest_contour(contours):
    # define empty variables
    biggest_polygonal_curve = numpy.array([])
    max_area = 0
    for cntr in contours:
        area = cv.contourArea(cntr)
        # skip very small areas
        if area > 5000:
            contour_perimeter = cv.arcLength(cntr, True)
            polygonal_curve = cv.approxPolyDP(cntr, 0.02 * contour_perimeter, True)
            # look for quadrilateral with the biggest area
            if len(polygonal_curve) == 4 and area > max_area:
                biggest_polygonal_curve = polygonal_curve
                max_area = area
    return biggest_polygonal_curve


def warp_image(image, contour):
    src = reorder_coordinates(contour)
    dst = [[0, 0], [width, 0], [0, height], [width, height]]
    matrix = cv.getPerspectiveTransform(numpy.float32(src), numpy.float32(dst))
    image_warp = cv.warpPerspective(image, matrix, (width, height))
    return image_warp


def reorder_coordinates(contour):
    src = numpy.resize(contour, (4, 2))
    array_sum_of_coordinates = numpy.sum(src, axis=1)
    array_sorted_sum_of_coordinates = numpy.argsort(array_sum_of_coordinates)
    array_sorted_coordinates = numpy.array(
        [src[array_sorted_sum_of_coordinates[0]],
         src[array_sorted_sum_of_coordinates[1]],
         src[array_sorted_sum_of_coordinates[2]],
         src[array_sorted_sum_of_coordinates[3]]])
    return array_sorted_coordinates


# image of the document on the table
img = cv.imread("resources/IMG_1074.JPG")
img = cv.resize(img, (width, height))
binary_image = prepare_image(img)
img_copy_1 = img.copy()
# detect document contour
document_contour = get_contour(binary_image)
# draw contours on the original image
cv.drawContours(img_copy_1, document_contour, -1, (0, 255, 0), 3)
img_copy_2 = img.copy()
warped_image = warp_image(img_copy_2, document_contour)

cv.imshow('Output', warped_image)
cv.waitKey(0)
cv.destroyAllWindows()
