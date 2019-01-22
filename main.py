
import time
import RPi.GPIO as GPIO
import threading

FWD = GPIO.HIGH
REV = GPIO.LOW

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

    def Motor_X_Move(self, Distance_mm, Direction):
        GPIO.output(DIR_X, Direction)
        for x in range(Distance_mm):
            if GPIO.input(STOP_X) != GPIO.LOW:
                GPIO.output(STEP_X, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(STEP_X, GPIO.LOW)
                time.sleep(delay)

    def Motor_Y_Move(self, Distance_mm, Direction):
        GPIO.output(DIR_Y, Direction)
        for x in range(Distance_mm):
            if GPIO.input(STOPtop_Y) != GPIO.LOW:
                GPIO.output(STEP_Y, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(STEP_Y, GPIO.LOW)
                time.sleep(delay)


MotorControl = MotorMovement()
#######################################
# Functions

#######################################
# Calibration
# Initial
# How many steps needed to move 1mm
start = time.time()
MotorControl.Motor_X_Move(SPR, FWD)
finish = time.time()
print("Time taken:", (finish - start))

start = time.time()
MotorControl.Motor_Y_Move(SPR, FWD)
finish = time.time()
print("Time taken:", (finish-start))

# Motor
#   motor on while sensor not active, continue
#   stop motor for that sensor
#   other motor on until other sensor activates
MotorThreadX = threading.Thread(target=MotorControl.Motor_X_Move(2000, FWD))
MotorThreadY = threading.Thread(target=MotorControl.Motor_Y_Move(2000, FWD))
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
