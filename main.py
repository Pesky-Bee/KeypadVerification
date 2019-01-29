
import time
import RPi.GPIO as GPIO
import numpy as np
import cv2
import math
# from imutils.object_detection import non_max_suppression
# import pytesseract
# import argparse

GPIO.setwarnings(False)

###########################################
# Setup
# Use GPIO number not pin number

#   Definitions
FWD = GPIO.HIGH
REV = GPIO.LOW

# Motor X
DIR_X = 2  # Direction Pin
STEP_X = 3  # Step Pin

# Motor Y
DIR_Y = 14  # Direction Pin
STEP_Y = 15  # Step Pin

# Stop Pin
STOPtop_Y = 25
STOPbottom_Y = 8
STOP_X = 7

# set delay time
delay = 0.0025

# steps per revolution
SPR = 200

# set as outputs
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR_X, GPIO.OUT)
GPIO.setup(STEP_X, GPIO.OUT)
GPIO.setup(DIR_Y, GPIO.OUT)
GPIO.setup(STEP_Y, GPIO.OUT)

# set as inputs
GPIO.setup(STOPtop_Y, GPIO.IN)
GPIO.setup(STOPbottom_Y, GPIO.IN)
GPIO.setup(STOP_X, GPIO.IN)

# Microstep Resolution
MODE_X = (17, 27, 22)
MODE_Y = (18, 23, 24)
GPIO.setup(MODE_X, GPIO.OUT)
GPIO.setup(MODE_Y, GPIO.OUT)
RES = {'Full': (0, 0, 0),
       'Half': (1, 0, 0),
       'Quar': (0, 1, 0),
       'Eigh': (1, 1, 0),
       'Sixt': (1, 1, 1)}
GPIO.output(MODE_X, RES['Half'])
GPIO.output(MODE_Y, RES['Half'])

# Camera
cap = cv2.VideoCapture(0)   # default webcam

#######################################
# Global Variables


#######################################
# Classes
class MotorMovement:

    # def __init__(self):
    #    super(MotorMovement, self).__init__()

    def Motor_X_Move_FWD(self, Distance_mm):
        GPIO.output(DIR_X, FWD)
        for x in range(Distance_mm):
            if GPIO.input(STOP_X) != GPIO.LOW:
                GPIO.output(STEP_X, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(STEP_X, GPIO.LOW)
                time.sleep(delay)

    def Motor_X_Move_REV(self, Distance_mm):
        GPIO.output(DIR_X, REV)
        for x in range(Distance_mm):
                GPIO.output(STEP_X, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(STEP_X, GPIO.LOW)
                time.sleep(delay)

    def Motor_Y_Move_FWD(self, Distance_mm):
        GPIO.output(DIR_Y, FWD)
        for x in range(Distance_mm):
            if GPIO.input(STOPtop_Y) != GPIO.LOW:
                GPIO.output(STEP_Y, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(STEP_Y, GPIO.LOW)
                time.sleep(delay)
                
    def Motor_Y_Move_REV(self, Distance_mm):
        GPIO.output(DIR_Y, REV)
        for x in range(Distance_mm):
            if GPIO.input(STOPbottom_Y) != GPIO.LOW:
                GPIO.output(STEP_Y, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(STEP_Y, GPIO.LOW)
                time.sleep(delay)


#######################################
# Functions
def scale(frame, percent=50):
    width = int(frame.shape[1]*percent/100)
    height = int(frame.shape[0]*percent/100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


#######################################
# Calibration
# Motor
MotorControl = MotorMovement()

MotorControl.Motor_X_Move_REV(2000)
MotorControl.Motor_Y_Move_FWD(2000)
# Both motors now at max positive positions and out of the way of camera

# Camera
# Camera take picture of keypad to know where numbers located + recognition? and the pattern for distance, segment them?
ret, frame = cap.read()  # read frame
frame75 = scale(frame, percent=75)
framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(framegray, 30, 200)
ret, thresh = cv2.threshold(framegray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# #####
### noise removal
##kernel = np.ones((3, 3), np.uint8)
##opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
### sure background area
##sure_bg = cv2.dilate(opening, kernel, iterations=3)
### Finding sure foreground area
##dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
##ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
### Finding unknown region
##sure_fg = np.uint8(sure_fg)
##unknown = cv2.subtract(sure_bg, sure_fg)
### Marker labelling
##ret, markers = cv2.connectedComponents(sure_fg)
### Add one to all labels so that sure background is not 0, but 1
##markers = markers+1
### Mark the region of unknown with zero
##markers[unknown == 255] = 0
##markers = cv2.watershed(frame75, markers)
##frame75[markers == -1] = [255, 0, 0]
# #####
### OR
# find contours in the binary image
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
#dilation = cv2.dilate(opening, kernel, iterations = 1)
cv2.imshow("image", opening)

contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours[1:]:
    # calculate moments for each contour
    M = cv2.moments(c)
    # calculate x,y coordinate of center
    # ensure no division by 0
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    # visulise and place center
    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(frame, "c", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # display the image
    cv2.imshow("Image", frame)
cv2.waitKey(0)
# cv2.imwrite("thresh.jpg", thresh)
# cv2.imwrite("frame.jpg", frame75)

# cv2.imshow('frame', edges)
# cv2.imshow('frame',frame) #show initial frame

# Use image to Calibrate from known length between pattern
#   calculate distance of projected pattern line length, and scale from actual
# calculateDistance(x1, y1, x2, y2)

# Find where each button is, assign to location in 2D array?

# Adjust distances, scale motor travel distance

######################################

# Read first prompt
#   Region of interest?
# ScreenRegion = cv2.selectROI("Region", framegray, False, False)
# Interpret what needs to be pressed
#   Get screen command

#   Lookup what command needs doing

# Go through sequence of required presses
#   Move to first button to be pressed

#   Press Button

######################################
# Cleanup
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()

#####################################
##
### NOTES!
##
### https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
### pip install pytesseract
### pip install imutils
##
### construct the argument parser and parse the arguments
##ap = argparse.ArgumentParser()
##ap.add_argument("-i", "--image", type=str, help="path to input image")
##ap.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
##ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
##                help="minimum probability required to inspect a region")
##ap.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for resized width")
##ap.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for resized height")
##ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
##args = vars(ap.parse_args())
##
### load the input image and grab the image dimensions
##image = frame75
##orig = image.copy()
##(origH, origW) = image.shape[:2]
##
### set the new width and height and then determine the ratio in change
### for both the width and height
##(newW, newH) = (args["width"], args["height"])
##rW = origW / float(newW)
##rH = origH / float(newH)
##
### resize the image and grab the new image dimensions
##image = cv2.resize(image, (newW, newH))
##(H, W) = image.shape[:2]
##
### define the two output layer names for the EAST detector model that
### we are interested in -- the first is the output probabilities and the
### second can be used to derive the bounding box coordinates of text
##layerNames = [
##    "feature_fusion/Conv_7/Sigmoid",
##    "feature_fusion/concat_3"]
##
### load the pre-trained EAST text detector
##print("[INFO] loading EAST text detector...")
##net = cv2.dnn.readNet(args["east"])
##
### construct a blob from the image and then perform a forward pass of
### the model to obtain the two output layer sets
##blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
##net.setInput(blob)
##(scores, geometry) = net.forward(layerNames)
##
### decode the predictions, then  apply non-maxima suppression to
### suppress weak, overlapping bounding boxes
##(rects, confidences) = decode_predictions(scores, geometry)
##boxes = non_max_suppression(np.array(rects), probs=confidences)
##
### initialize the list of results
##results = []
##
### loop over the bounding boxes
##for (startX, startY, endX, endY) in boxes:
##    # scale the bounding box coordinates based on the respective
##    # ratios
##    startX = int(startX * rW)
##    startY = int(startY * rH)
##    endX = int(endX * rW)
##    endY = int(endY * rH)
##
##    # in order to obtain a better OCR of the text we can potentially
##    # apply a bit of padding surrounding the bounding box -- here we
##    # are computing the deltas in both the x and y directions
##    dX = int((endX - startX) * args["padding"])
##    dY = int((endY - startY) * args["padding"])
##
##    # apply padding to each side of the bounding box, respectively
##    startX = max(0, startX - dX)
##    startY = max(0, startY - dY)
##    endX = min(origW, endX + (dX * 2))
##    endY = min(origH, endY + (dY * 2))
##
##    # extract the actual padded ROI
##    roi = orig[startY:endY, startX:endX]
##
### in order to apply Tesseract v4 to OCR text we must supply
### (1) a language, (2) an OEM flag of 4, indicating that the we
### wish to use the LSTM neural net model for OCR, and finally
### (3) an OEM value, in this case, 7 which implies that we are
### treating the ROI as a single line of text
##config = "-l eng --oem 1 --psm 7"
##text = pytesseract.image_to_string(roi, config=config)
##
### add the bounding box coordinates and OCR'd text to the list
### of results
##results.append(((startX, startY, endX, endY), text))
##
### sort the results bounding box coordinates from top to bottom
##results = sorted(results, key=lambda r: r[0][1])
##
### loop over the results
##for ((startX, startY, endX, endY), text) in results:
##    # display the text OCR'd by Tesseract
##    print("{}\n".format(text))
##
##    # strip out non-ASCII text so we can draw the text on the image
##    # using OpenCV, then draw the text and a bounding box surrounding
##    # the text region of the input image
##    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
##    output = orig.copy()
##    cv2.rectangle(output, (startX, startY), (endX, endY),
##                  (0, 0, 255), 2)
##    cv2.putText(output, text, (startX, startY - 20),
##                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
##
##    cv2.waitKey(0)
