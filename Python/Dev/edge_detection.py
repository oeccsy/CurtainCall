import cv2 as cv
import numpy as np

# Initialize control parameters
threshold1 = 500
threshold2 = 150
aperture_size = 3

while True:
    # Read the given image
    img = cv.imread("./carrot.png", cv.IMREAD_GRAYSCALE)
    
    assert img is not None, 'Cannot read the given image, ' + "./carrot.png"

    # Get the Canny edge image
    edge = cv.Canny(img, threshold1, threshold2, apertureSize=aperture_size)

    cv.imshow('Canny Edge', edge)

    # Process the key event
    key = cv.waitKey()
    if key == 27: # ESC
        break

cv.destroyAllWindows()
