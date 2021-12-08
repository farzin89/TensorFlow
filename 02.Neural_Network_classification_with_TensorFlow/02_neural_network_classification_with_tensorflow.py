# -*- coding: utf-8 -*-
"""02_Neural_Network_classification_with_TensorFlow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IMq3E7gkcBE9CtLXrKA9C9wcPWemQnJQ

## Itroduction to neural network classification with TensorFlow

#### in this notebook we're going to learn how to write neural networks for classification problems.

### A classification is where you try to classify something as one thing or another .

### A few type of classication problems:    

#### Binary classification 
#### Multiclass classification
#### Multiable classification

## Creating data to view and fit
"""

from sklearn.datasets import make_circles

# Make 1000 example 
n_samples = 1000
# create circles 

x,y = make_circles(n_samples,
                   noise = 0.03,
                   random_state =42)

# check out features 
x

# check the labels
y[:10]

"""#### our data is a little hard to underestand right now... let's visualize it"""

import tensorflow as tf
import pandas as pd 
circles = pd.DataFrame({"x0":x[:,0],"x1":x[:,1], "label":y})
circles

circles["label"].value_counts()

# Visualize with a plot 

import matplotlib.pyplot as plt
plt.scatter(x[:,0],x[:,1],c=y, cmap= plt.cm.RdYlBu);

"""### Exercise: Before pushing forward, spend 10-minutes playing with  https://playground.tensorflow.org/ bulding and running different neural networks. See what happens when you change different hyperparameters.

## Input and output shapes
"""

# check the shapes of our features and labels 
x.shape, y.shape

# how many samples we're working 
len(x),len(y)

# View the first example of feature and labels 
x[0],y[0]

"""## Steps in modelling  

### The steps in modelling with TensorFlow are typically:   

#### 1. Create or import a model 
#### 2. compile the model 
#### 3. Fit the model 
#### 4. Evaluate the model 
#### 5. Tweak 
#### 6. Evaluate

"""

# set the random seed

tf.random.set_seed(42)

# 1 .Create the model using the Sequential API

model_1 = tf.keras.Sequential([
      tf.keras.layers.Dense(1)
])

# 2. compile the model 
model_1.compile(loss= tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.SGD(),
                metrics= ["accuracy"])
# 3. Fit the model 

model_1.fit(x,y,epochs=5 )

# lets try and improve our model by training for longer...

model_1.fit(x,y,epochs=200, verbose=0)
model_1.evaluate(x,y)

"""### since we're working on a binary on a classification problem and our model is getting around  50 % accuracy... it's performing as if it's gussing.

#### so lets things up a notch and add an extra layer.
"""

# set the random seet

tf.random.set_seed(42)

# 1. Create a model , this time with 2 layers

model_2 = tf.keras.Sequential([tf.keras.layers.Dense(1),
                              tf.keras.layers.Dense(1),
])
# 2. compile the model 

model_2.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer= tf.keras.optimizers.SGD(),
                metrics = ["accuracy"]
                )
# 3. fit the model 

model_2.fit(x,y,epochs = 100, verbose=0)

# evaluate the model 

model_2.evaluate(x,y)

circles["label"].value_counts()

"""## imrove our model

### let's look our bag of tricks to see how we can improve our model. 
 1. creat a model - we might to add more layers or increase the number of hidden units within a layer. 
 2. compiling a model - here we might to choose a diferent optimization function such as adam instead of SGD
 3. fiting a model - perhaps we might fit our model for more epochs (leave it training for longer).
  

"""

# set the random seed

tf.random.set_seed(42)

# 1. create  the model (this time 3 layers)

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100), # add 100 dense neurons    
    tf.keras.layers.Dense(10),  # add another layer with 10 neurons 
    tf.keras.layers.Dense(1)

])

# 2. compile the model 

model_3.compile( loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer= tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

# 3. fit the model 

model_3.fit(x,y,epochs=100 ,verbose = 0 )

# 4. Evaluate the model 
model_3.evaluate(x,y)



"""#### To visulize our model's predictions,let's create a function 'plot_decision_boundary()' this function will :    
 ### - Take in a trained model,features(x) and labels(y)
 ### - Create a meshgred of the different x values
 ### - Make predictions across the meshgrid 
 ### - plot the predictions as well as a line between zones (where each unique class falls).
"""

import numpy as np

import numpy as np

def plot_decision_boundary(model, x, y):
  """
  Plots the decision boundary created by a model predicting on X.
  This function has been adapted from two phenomenal resources:
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
  """
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
  y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  
  # Create X values (we're going to predict on all of these)
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html
  
  # Make predictions using the trained model
  y_pred = model.predict(x_in)

  # Check for multi-class
  if len(y_pred[0]) > 1:
    print("doing multiclass classification...")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(y_pred).reshape(xx.shape)
  
  # Plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

# cheak out the predictions our model is making 
plot_decision_boundary(model= model_3,x=x , y=y)



# Set random seed
tf.random.set_seed(42)

# Create some regression data
X_regression = np.arange(0, 1000, 5)
y_regression = np.arange(100, 1100, 5)

# Split it into training and test sets
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]
y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]

# Fit our model to the data
# Note: Before TensorFlow 2.7.0, this line would work
# model_3.fit(X_reg_train, y_reg_train, epochs=100)

# After TensorFlow 2.7.0, see here for more: https://github.com/mrdbourke/tensorflow-deep-learning/discussions/278
model_3.fit(tf.expand_dims(X_reg_train, axis=-1), 
            y_reg_train,
            epochs=100)

"""## oh wait... we comiled our model for a binary classification problem . But .... we're now working on a regression problem, lets change the model to suit our data."""

# setup random seed
tf.random.set_seed(42)
# create the model 

model_3 = tf.keras.Sequential([
          tf.keras.layers.Dense(100),
          tf.keras.layers.Dense(10),
          tf.keras.layers.Dense(1)                     
])

# 2. Compile the model, this time with a regression - specific loss function 
model_3.compile(loss = tf.keras.losses.mae,
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["mae"])
# fit the model

model_3.fit(x_reg_train,y_reg_train,epochs=100)

# Set random seed
tf.random.set_seed(42)

# Create some regression data
x_regression = np.arange(0, 1000, 5)
y_regression = np.arange(100, 1100, 5)

# Split it into training and test sets
x_reg_train = x_regression[:150]
x_reg_test = x_regression[150:]
y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]

# Fit our model to the data
# Note: Before TensorFlow 2.7.0, this line would work
# model_3.fit(X_reg_train, y_reg_train, epochs=100)

# After TensorFlow 2.7.0, see here for more: https://github.com/mrdbourke/tensorflow-deep-learning/discussions/278
model_3.fit(tf.expand_dims(x_reg_train, axis=-1), 
            y_reg_train,
            epochs=100)

# Make predictions with our trained model
y_reg_preds = model_3.predict(x_reg_test)

# Plot the model's predictions against our regression data
plt.figure(figsize=(10, 7))
plt.scatter(x_reg_train, y_reg_train, c='b', label='Training data')
plt.scatter(x_reg_test, y_reg_test, c='g', label='Testing data')
plt.scatter(x_reg_test, y_reg_preds, c='r', label='Predictions')
plt.legend();

"""  ### The missing piece  : Non-linearity"""

# Set the random seed()

tf.random.set_seed(40)

# 1.Create the model 
model_4 = tf.keras.Sequential([
          tf.keras.layers.Dense(1, activation= tf.keras.activations.linear),
          tf.keras.layers.Dense(10),
          tf.keras.layers.Dense(1)                               
])

# 2.compile the model 
model_4.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer =tf.keras.optimizers.Adam(lr=0.001),
                metrics = ["accuracy"])

# 3. fit the model 
history = model_4.fit(x,y,epochs=100)

# Check out our data 


plt.scatter(x [:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu);

# check the decision boundary for our latest model 

plot_decision_boundary(model_4,x=x ,y=y)

"""### Let's try build our first neural network with a non-linear activation function.

"""

# Set radom seed 
tf.random.set_seed(42)
# 1. Create the model  with a non_linear activation 

model_5 = tf.keras.Sequential([ 
          tf.keras.layers.Dense(1, activation= tf.keras.activations.relu)
                                                    
])

# 2. compile the model 

model_5.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(lr=0.001),
                metrics= ["accuracy"])
# 3.fit the model 
model_5.fit(x,y,epochs= 100)

# Time to replicate the multi-layer neural network from TensorFlow playground in code ...

# set the random seed 
tf.random.set_seed(42)
# 1. Create the model 
model_6 = tf.keras.Sequential([
          tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
          tf.keras.layers.Dense(4,activation=tf.keras.activations.relu),
          tf.keras.layers.Dense(4,)
                                ])
# 2. compile the model 

model_6.compile (loss = tf.keras.losses.binary_crossentropy,
                 optimizer = tf.keras.optimizers.Adam(lr = 0.001),
                 metrics = ["accuracy"]
                 )

# 3. fit the model 

model_6.fit(x,y,epochs= 250)

# Evaluate the model 
model_6.evaluate(x,y)

# how do our model predictions look?
plot_decision_boundary(model_6,x,y)

# set the random seed 
tf.random.set_seed(42)

# 1. Create the models 

model_7 = tf.keras.Sequential([tf.keras.layers.Dense(4, activation= "relu"),
                               tf.keras.layers.Dense(4, activation= "relu"),
                               tf.keras.layers.Dense(1, activation="sigmoid")
                               ])
# 2. compile the model 

model_7.compile(loss = tf.keras.losses.binary_crossentropy,
                optimizer = tf.keras.optimizers.Adam(lr=0.001),
                metrics = ["accuracy"])
# 3. fit the model 

model_7.fit(x,y, epochs = 100)

# 4. Evaluate the model 

model_7.evaluate(x,y)

# Lets visualize our increadible metrics 

plot_decision_boundary(model_7,x,y)

"""### Question : what's wrong the predictions we've made? Are we really evaluating our model correctly? hint: what data did the model learn on and what data did we predict on?

### Note : The combination of linear (straight lines) and non-linear(non-straight lines) functions is one of the key fundamentals of neural networks.

### Now we've discussed the concept of linear and non=linear functions (or lines), let's see them in sction.
"""

# Create a toy tensor (similar to the data we pass into our models)

  A = tf.cast(tf.range(-10,10),tf.float32)
  A

plt.plot(A)

# Let's start by replicating sigmoid ,  sigmoid(x)=1/(1+exp(-x))
def sigmoid(x):
   return 1/(1+tf.exp(-x))

# Use the sigmoid function on our toy tensor
sigmoid(A)

# Plot our toy tensor transformed by sigmoid 

plt.plot(sigmoid(A))

# L3t's recreate the relu function

def relu(x):
  return tf.maximum(0,x)
# pass our toy tensor to our custom relue function
relu (A)

A

# plot ReLU-modified tensor
plt.plot(relu(A))

# Let's try the linear activation function 
tf.keras.activations.linear(A)

plt.plot(tf.keras.activations.linear(A))

# Does A even change?

A == tf.keras.activations.linear(A)

""" ## Evaluating and improving our classification

 #### so far we've been training and testing on the same dataset.. However,in machine learning this is basically a sin.

 #### let's create a teraining and test set.
"""

# Check how many examples we have 
len(x)

x,y

# Split into train and test sets 
x_train,y_train = x[:800],y[:800]
x_test,y_test = x[800:] , y[800:]

x_train.shape,x_test.shape,y_train.shape,y_test.shape

# let's recreate a model to fit on the training data and evaluate on the testing data

# set random seed
tf.random.set_seed(42)

# 1. create the model 

model_8 = tf.keras.Sequential([tf.keras.layers.Dense(4, avtivations)])