from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time

import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)

#%matplotlib inline

file_dataTrain = "./data_train.csv"
file_dataTest = "./data_test.csv"

# Training parameters
BATCH_SIZE = 100
EPOCHS = 2000
time_steps = 30

df_train = pd.read_csv(file_dataTrain)
df_test = pd.read_csv(file_dataTest)

def generate_data(data, time_steps):
    """
    generate sequences
    """
    rnn_x = []
    for i in range(len(data) - time_steps + 1): 
        rnn_x.append(data[['PRCP', 'TAVG', 'count(t-1)']].iloc[i:i + time_steps].as_matrix())

    rnn_x = np.array(rnn_x)
    rnn_y = data[['count']][time_steps - 1:].values
    return rnn_x, rnn_y

trainX, trainY = generate_data(df_train, time_steps)
testX, testY = generate_data(df_test, time_steps)

def create_model(x):
    """
    Create the model for prediction
    """

    with C.layers.default_options(initial_state = 0.1):
        m = C.layers.Recurrence(C.layers.LSTM(200))(x)
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2, seed = 1)(m)
        m = C.layers.Dense(1)(m)
        return m

def next_batch(x, y):
    """
    Get the next batch to process
    """
    def as_batch(data, start, count):
        part = []
        for i in range(start, start + count):
            part.append(data[i])
        return np.array(part)

    for i in range(0, len(x) - BATCH_SIZE, BATCH_SIZE):
        yield as_batch(x, i, BATCH_SIZE), as_batch(y, i, BATCH_SIZE)

# input sequences
x = C.sequence.input_variable(3)

# create the model
z = create_model(x)

# expected output (label), also the dynamic axes of the model output
# is specified as the model of the label input
l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")

# the learning rate
learning_rate = 0.02
lr_schedule = C.learning_parameter_schedule(learning_rate)

# loss function
loss = C.squared_error(z, l)

# use squared error to determine error for now
error = C.squared_error(z, l)

# use fsadagrad optimizer
momentum_schedule = C.momentum_schedule(0.9, minibatch_size=BATCH_SIZE)
learner = C.fsadagrad(z.parameters, 
                      lr = lr_schedule, 
                      momentum = momentum_schedule, 
                      unit_gain = True)

trainer = C.Trainer(z, (loss, error), [learner])

# Train
loss_summary = []
start = time.time()
for epoch in range(0, EPOCHS):
    for x1, y1 in next_batch(trainX, trainY):
      #  print("x1 = ", x1)
      #  print("y1 = ", y1)
        trainer.train_minibatch({x: x1, l: y1})
    if epoch % (EPOCHS / 10) == 0:
        training_loss = trainer.previous_minibatch_loss_average
        loss_summary.append(training_loss)
        print("epoch: {}, loss: {:.5f}".format(epoch, training_loss))

print("training took {0:.1f} sec".format(time.time() - start))
# A look how the loss function shows how well the model is converging
plt.figure(1)
plt.subplot(211)
plt.plot(loss_summary, label='training loss')

# validate
def get_mse(X,Y):
    result = 0.0
    for x1, y1 in next_batch(X, Y):
        eval_error = trainer.test_minibatch({x : x1, l : y1})
        result += eval_error
    return result/len(X)

# Print the train/test errors
print("train mse: {:.6f}".format(get_mse(trainX, trainY)))
print("test mse: {:.6f}".format(get_mse(testX, testY)))

# predict
results = []
for x1, y1 in next_batch(testX, testY):
    pred = z.eval({x: x1})
    results.extend(pred[:, 0])

plt.subplot(212)
plt.plot(testY, label = 'test data')
plt.plot(results, label = 'test predicted')
plt.show()
