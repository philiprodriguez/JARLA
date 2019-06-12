import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib
from matplotlib import pyplot
from matplotlib import image
import jarla_env
from jarla_env import JarlaEnvironment
import time
print("JARLA is starting...")

# Build the JARLA neural network model...

input_layer = layers.Input(shape=(JarlaEnvironment.CONST_IMAGE_HEIGHT, JarlaEnvironment.CONST_IMAGE_WIDTH, 3))
x = layers.Conv2D(8, kernel_size=(5, 5), padding='valid', activation='relu')(input_layer)
x = layers.Conv2D(8, kernel_size=(5, 5), padding='valid', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
x = layers.Conv2D(8, kernel_size=(3, 3), padding='valid', activation='relu')(x)
x = layers.Conv2D(8, kernel_size=(3, 3), padding='valid', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
x = layers.Conv2D(8, kernel_size=(3, 3), padding='valid', activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
x = layers.Conv2D(4, kernel_size=(3, 3), padding='valid', activation='relu')(x)

x = layers.Flatten()(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(len(JarlaEnvironment.CONST_ACTIONS), activation='linear')(x)

model = Model(inputs=input_layer, outputs=x)

print(model.summary())

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# image = image.imread("image.jpg")
# print(image.shape)
# pyplot.imshow(image)
# pyplot.show()

# Construct environment instance
env = JarlaEnvironment()

y = 0.80
eps = 0.5
eps_decay_factor = 0.9999
max_run_iterations = 99999
reward_history = []
for i in range(1, max_run_iterations+1):
    eps *= eps_decay_factor

    print("Iteration " + str(i) + " of " + str(max_run_iterations))
    print("Epsilon is at " + str(eps))

    # What do we think of our current state?
    iteration_start_state = env.get_state()
    start_time = time.time()
    q_function_result = model.predict(iteration_start_state)
    end_time = time.time()
    print("Prediction took " + str(end_time-start_time) + "s")

    # Select some action to take
    if np.random.random() < eps:
        # Select a random action
        action_selection = np.random.randint(0, len(JarlaEnvironment.CONST_ACTIONS))
        print("Selected random action of " + str(action_selection))
    else:
        # Act according to what we think will give best reward
        action_selection = np.argmax(q_function_result[0])
        print("Selected action of " + str(action_selection) + " based on " + str(q_function_result[0]))

    # Perform the action
    reward = env.act(action_selection)
    reward_history.append(reward)
    print("Recieved reward of " + str(reward))
    jarla_env.set_led(reward)

    # Compute the reward seemingly gained by taking the action
    iteration_end_state = env.get_state()
    perceived_reward = reward + y * np.max(model.predict(iteration_end_state))
    print("Perceived reward is " + str(perceived_reward))

    # Prepare a "correct answer" vector for the neural network
    perceived_reward_train_vec = np.copy(q_function_result)
    perceived_reward_train_vec[0][action_selection] = perceived_reward

	# Train on correct answer!
    model.fit(x=iteration_start_state, y=perceived_reward_train_vec, batch_size=1, epochs=1)
print(reward_history)
