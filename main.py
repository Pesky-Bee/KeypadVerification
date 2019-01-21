
from time import sleep
import RPi.GPIO as GPIO
import threading
from queue import Queue

###########################################
# Setup
# Use GPIO number not pin number
# Motor X
DIR_X = 2  # Direction Pin
STEP_X = 3  # Step Pin

# Motor Y
DIR_Y = 14  # Direction Pin
STEP_Y = 15  # Step Pin

# Stop Pin
STOPtop_Y = 8
STOPbottom_Y = 7
STOP_X = 25

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


#######################################
# Classes
class MotorMovement(threading.Thread):

    def __init__(self):
        super(MotorMovement, self).__init__()

    def Motor_X_Move_FWD(self, Distance_mm):
        for x in range(Distance_mm):
            if GPIO.input(STOP_X) != GPIO.LOW:
                GPIO.output(DIR_X, GPIO.HIGH)
                GPIO.output(STEP_X, GPIO.HIGH)
                sleep(delay)
                GPIO.output(STEP_X, GPIO.LOW)
                sleep(delay)

    def Motor_X_Move_REV(self, Distance_mm):
        for x in range(Distance_mm):
            GPIO.output(DIR_X, GPIO.LOW)
            GPIO.output(STEP_X, GPIO.HIGH)
            sleep(delay)
            GPIO.output(STEP_X, GPIO.LOW)
            sleep(delay)

    def Motor_Y_Move_FWD(self, Distance_mm):
        for x in range(Distance_mm):
            if GPIO.input(STOPtop_Y) != GPIO.LOW:
                GPIO.output(DIR_Y, GPIO.HIGH)
                GPIO.output(STEP_Y, GPIO.HIGH)
                sleep(delay)
                GPIO.output(STEP_Y, GPIO.LOW)
                sleep(delay)

    def Motor_Y_Move_REV(self, Distance_mm):
        for x in range(Distance_mm):
            if GPIO.input(STOPbottom_Y) != GPIO.LOW:
                GPIO.output(DIR_Y, GPIO.LOW)
                GPIO.output(STEP_Y, GPIO.HIGH)
                sleep(delay)
                GPIO.output(STEP_Y, GPIO.LOW)
                sleep(delay)


#######################################
# Functions

#######################################
# Calibration
# Initial
# How many steps needed to move 1mm
start = time.time
MotorMovement.Motor_X_Move_FWD(self, SPR)
finish = time.time
print("Time taken:", (finish - start))

start = time.time
MotorMovement.Motor_Y_Move_FWD(self, SPR)
finish = time.time
print("Time taken:", (finish - start))

# Motor
#   motor on while sensor not active, continue
#   stop motor for that sensor
#   other motor on until other sensor activates
MotorThreadX = threading.Thread(target=Motor_X_FWD, args=(self, 200))
MotorThreadY = threading.Thread(target=Motor_Y_FWD, args=(self, 200))
MotorThreadX.start()
MotorThreadY.start()
MotorThreadX.join()
MotorThreadY.join()
# both motors now at max positive positions


# Camera
#   calculate distance of projected pattern line length, and
#   scale from actual
#   find where each button is, assign to location in array?

#   adjust distances, scale motor travel distance

# Motors to 'home'

# Camera take picture of keypad to know where numbers located
######################################

# Read first prompt

# Interpret what needs to be pressed

# Go through sequence of required presses
#   Move to first button to be pressed

GPIO.cleanup()

# threading.Thread(target=Motor_X_FWD, args=(self,10)).start()
# threading.Thread(target=Motor_X_REV, args=(self,10)).start()
# threading.Thread(target=Motor_Y_FWD, args=(self,10)).start()
# threading.Thread(target=Motor_Y_REV, args=(self,10)).start()
