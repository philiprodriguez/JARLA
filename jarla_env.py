# This module contains all code relevant to interfacing with the environment
# of JARLA.

import numpy as np
import threading
import time
import RPi.GPIO as GPIO
import matplotlib
from matplotlib import pyplot
from matplotlib import image
from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (256, 256)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(256, 256))
time.sleep(0.2)

def get_frame():
    camera.capture(rawCapture, format='rgb', use_video_port=True)
    image = np.copy(rawCapture.array)
    rawCapture.truncate(0)
    return image

#while True:
#    start_time = time.time()
#    camera.capture(rawCapture, format='rgb', use_video_port=True)
#    image = rawCapture.array
#    end_time = time.time()
#    print("Took " + str(end_time-start_time) + "s")
#    pyplot.imshow(image)
#    pyplot.show()
#    rawCapture.truncate(0)


pin_led = 11
pin_rmf = 18
pin_rmb = 16
pin_lmf = 13
pin_lmb = 15

GPIO.setmode(GPIO.BOARD)
GPIO.setup(pin_led, GPIO.OUT)
GPIO.setup(pin_rmf, GPIO.OUT)
GPIO.setup(pin_rmb, GPIO.OUT)
GPIO.setup(pin_lmf, GPIO.OUT)
GPIO.setup(pin_lmb, GPIO.OUT)

led_pwm = GPIO.PWM(pin_led, 1)
led_pwm.start(50.0)

def motors_off():
    GPIO.output(pin_rmf, False)
    GPIO.output(pin_rmb, False)
    GPIO.output(pin_lmf, False)
    GPIO.output(pin_lmb, False)
motors_off()

def set_led(reward):
    led_pwm.ChangeFrequency(min(reward/6.375, 40.0))

def thread_action_forward():
    # Instruct motors
    GPIO.output(pin_lmf, True)
    GPIO.output(pin_rmf, True)

    # Wait one second
    time.sleep(1.0)

    # Instruct motors
    motors_off()

def thread_action_backward():
    # Instruct motors
    GPIO.output(pin_lmb, True)
    GPIO.output(pin_rmb, True)

    # Wait one second
    time.sleep(1.0)

    # Instruct motors
    motors_off()

def thread_action_left():
    # Instruct motors
    GPIO.output(pin_lmb, True)
    GPIO.output(pin_rmf, True)

    # Wait one second
    time.sleep(0.5)

    # Instruct motors
    motors_off()

def thread_action_right():
    # Instruct motors
    GPIO.output(pin_lmf, True)
    GPIO.output(pin_rmb, True)

    # Wait one second
    time.sleep(0.5)

    # Instruct motors
    motors_off()

class JarlaEnvironment:
    CONST_IMAGE_WIDTH = 256
    CONST_IMAGE_HEIGHT = 256
    CONST_ACTIONS = [
        thread_action_forward,
        thread_action_backward,
        thread_action_left,
        thread_action_right
    ]

    def __init__(self):
        pass

    # Return the state of the environment!
    def get_state(self):
        return np.reshape(get_frame(), (1, 256, 256, 3))/255.0
        
    def get_current_reward(self):
        frame = get_frame()
        return np.mean(frame)

    def act(self, action_number):
        t = threading.Thread(target=self.CONST_ACTIONS[action_number])
        t.start()
        t.join()
        return self.get_current_reward()
        
#    # To be used in the future to pipeline
#    def act_and_fit(self, action_number, model, iteration_start_state, perceived_reward_train_vec):
#        # Begin perform action
#        model.fit(x=iteration_start_state, y=perceived_reward_train_vec, batch_size=1, epochs=1)
#        
#        # Join on action completion
#        # Return reward
