# Import keras modules
from keras.models import Sequential
from keras.layers import Dense

# Import numpy to define our arrays
import numpy as np

# Inputs: This is out input numpy array, consisting or three column inputs and
# 7 rows of training data
predictors = np.array([[0, 0, 1],
                       [0, 1, 1],
                       [1, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0],
                       [1, 1, 1],
                       [0, 0, 0]])

# Outputs: Our model will try to optimize neuron weights to reach the output below for each
# array of our training data. The T at the end is to transpose the data from a row
# to a column
target = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

#  Tell Keras we want to use the Sequential model to build our network
model = Sequential()

# Add our first hidden network layer with which is Dense, meaning all layers in previous layer are
# connected to all layers in current layer
# We use 100 neurons and 3 inputs columns.
# Using 'tanh' as our activation function, 'relu' and 'sigmoid' are also popular
# You can start with a lower number of neurons and using trial and error by adjusting
# the number until you see little improvement in loss and accuracy
model.add(Dense(300, activation='tanh', input_shape=(3,)))
# Adding our second hidden layer.  Turn it off along with other layer and see how that
# affects your results
model.add(Dense(100, activation='tanh'))
# Our third hidden layer
model.add(Dense(50, activation='tanh'))
# Our fourth hidden layer
model.add(Dense(25, activation='tanh'))
# Our output layer, has one neuron, we are expecting a single number for each of
# our 7 input arrays
model.add(Dense(1, activation='sigmoid'))

# Compile our model using the following optimizer and loss functions.
# optimizer: controls the learning rate at which the model adjusts for optimal weights
# SGD(Stochastic Gradient Descent) and relu are also a commonly used optimizers
# loss: controls the function for measuring the error, a lower error rate is better
# MSE or Mean Squared Error is another commonly used loss function
# metrics will print out our model's accuracy during compilation
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Our model is now ready to run and will try and adjust weights using backpropagation
# epochs: 20,000 iterations
# epochs tells Keras how many times do we want to run training iterations
# Start will low number of epochs and adjust until you see no apparent increase
# in accuracy or decrease in loss
model_output = model.fit(predictors, target, epochs=20000)

# Now let's use out trained model to predict the output for [1,1,0]
# which should give us a value of 0
test_array = np.array([[1, 1, 0]])
predictions = model.predict(test_array)
print('-'*30)
print('Predicted output for [1,1,0]')
print('{:.10f}'.format(predictions[0][0]))

# Let's see the output our model gives us for the inputs we provided to train it
print('-'*30)
print('Predictions for supplied inputs using our new model')
print(model.predict_proba(predictors))





