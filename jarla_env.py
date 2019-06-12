# This module contains all code relevant to interfacing with the environment
# of JARLA.

import numpy as np

class JarlaEnvironment:
    CONST_IMAGE_WIDTH = 512
    CONST_IMAGE_HEIGHT = 512
    CONST_NUMBER_OF_ACTIONS = 4

    def __init__(self):
        pass

    # Return the state of the environment!
    def get_state(self):
        # For now, return a random 3-channel 512x512 image, normalized.
        return np.random.randint(0, 256, (1, 512, 512, 3))/255.0

    def act(self, action_number):
        # For now, return a random reward
        return np.random.randint(0, 10)
