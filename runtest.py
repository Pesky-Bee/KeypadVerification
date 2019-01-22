import time
import RPi.GPIO as GPIO

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


start = time.time()
GPIO.output(DIR_X, GPIO.HIGH)
GPIO.output(STEP_X, GPIO.HIGH)
while STOP_X != GPIO.LOW:
    continue
GPIO.output(STEP_X, GPIO.LOW)
print(time.time()-start)
# time / distance = delay
