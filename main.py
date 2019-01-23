
import time
import RPi.GPIO as GPIO
import numpy as np
import cv2
import math

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

    def __init__(self):
        super(MotorMovement, self).__init__()

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


MotorControl = MotorMovement()


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


#######################################
# Calibration
# Motor
MotorControl.Motor_X_Move_REV(2000)
MotorControl.Motor_Y_Move_FWD(2000)
# Both motors now at max positive positions and out of the way of camera

# Camera
# Camera take picture of keypad to know where numbers located and the pattern for distance, segment them?
ret, frame = cap.read()  # read frame
frame75 = scale(frame, percent=75)
framegray = cv2.cvtColor(frame75, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(framegray, 30, 200)
ret, thresh = cv2.threshold(framegray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Mark the region of unknown with zero
markers[unknown == 255] = 0
markers = cv2.watershed(frame75, markers)
frame75[markers == -1] = [255, 0, 0]

# cv2.imshow('frame',framegray)
# cv2.imshow('frame', edges)
# cv2.imshow('frame',frame) #show frame

# Use image to Calibrate from known length between pattern
#   calculate distance of projected pattern line length, and scale from actual
calculateDistance(x1, y1, x2, y2)

# Find where each button is, assign to location in 2D array?

# Adjust distances, scale motor travel distance

######################################

# Read first prompt

# Interpret what needs to be pressed

# Go through sequence of required presses
#   Move to first button to be pressed

#   Press Button


# Cleanup
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()
