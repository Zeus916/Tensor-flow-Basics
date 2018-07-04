'''
This is a basic implementation of a regression model where the model learns
to best fit a line in the linearly distributed data by learning over the errors made and 
correcting it.
This is over a 1000 iterations here in the code

'training_steps' can be changed to alter the training iterations.


'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)   # create some linearly spaced data and add some noise to it..
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

plt.subplot(121)
plt.title('Data')
plt.plot(x_data,y_label, '*')

m = tf.Variable(0.44)
b = tf.Variable(0.87)

error  = 0

for x,y in zip(x_data, y_label):
    y_hat = m*x + b                # yhat is the predicted value
    error += (y-y_hat)**2          # Compute the Quadratic loss function... summation(y -  predicted)**2
    # we want to minimize the error
    # th error gets more visibility on squaring..

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()  # initialize the tensorflow variables m and b that were decalred..

with tf.Session() as sess:
    sess.run(init)
    training_steps = 1000
    for i in range (training_steps):
        sess.run(train)
    final_slope, final_intercept = sess.run([m,b]) 

# testing...


x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept   # y = calculated slope * test_input + calculated intercept

#plot the output
plt.subplot(122)
plt.title('Linear Fit')
plt.plot(x_test, y_pred_plot, 'red')
plt.plot(x_data,y_label, '*') 
plt.show()