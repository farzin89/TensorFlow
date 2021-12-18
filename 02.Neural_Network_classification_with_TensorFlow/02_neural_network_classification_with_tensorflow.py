
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

model_8 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# 2. comile the model

model_8.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.01),
                metrics=["accuracy"])
# 3. Fit the model
history = model_8.fit(x_train, y_train, epochs=25)

# 4. Evaluate the model on test dataset

model_8.evaluate(x_test, y_test)

# plot the decision boundaries for the training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_8, x=x_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_8, x=x_test, y=y_test)
plt.show()

"""### plot the loss(or training) curves 


"""

# Convert the history object into a DataFrame

pd.DataFrame(history.history)

# plot the loss curves
pd.DataFrame(history.history).plot()
plt.title("Model_8 loss curves")

"""### Note : For many problems, the loss function going down means the model is improving ( the prediction it's making are getting ground truth labels).

## Finfing the best learning rate 

 ### 1. A learning rate "Callback" - you can think of a callback as an extra piece of funtionality , you can add to your while it's training.
 ### 2. Another model (we could use the same one as above, but we're practicing building models here) 
 ### 3. a modified loss curves plot.
"""

# set random seed

tf.random.set_seed(42)

# 1. Create the model (same as model_8 )

model_9 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# 2. compile the model
model_9.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer="Adam",
                metrics=["accuracy"])

# create a learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 20))
# 3. Fit the model(passing lr_schedular callback)
history_9 = model_9.fit(x_train,
                        y_train,
                        epochs=100,
                        callbacks=[lr_scheduler])

# checkout the history
pd.DataFrame(history_9.history).plot(figsize=(10, 7), xlabel="epochs")

# plot the learning rate versus the loss
lrs = 1e-4 * (10 ** (tf.range(100) / 20))
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history_9.history["loss"])
plt.xlabel("Learning Rate ")
plt.ylabel("Loss")
plt.title("Learning rate vs. Loss")

# Example of other typical learning rate values:

10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 1e-4

# Learning rate we used before (model_8)
10 ** -2

# Let's try using a higher "ideal" learning rate with the same model as before

# Set random seed

tf.random.set_seed(42)

# 1. create the model

model_10 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# 2. compile the model with ideal learning rate
model_10.compile(loss=tf.keras.losses.binary_crossentropy,
                 optimizer=tf.keras.optimizers.Adam(lr=0.02),
                 metrics=["accuracy"])
# 3. Fit the model for 20 epochs (5 less than before)
history_10 = model_10.fit(x_train, y_train, epochs=20)

# Evaluate model_10 on the test dataset
model_10.evaluate(x_test, y_test)

# Evaluate model_8 on the test dataset
model_8.evaluate(x_test, y_test)

# plot the decision boundaries for the training and test sets

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_10, x=x_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_10, x=x_test, y=y_test)
plt.show()

"""## More classification evaluation methods

### Alongside visualizing our models results as much as possible , there are a handful of other classification evaluation methods & metrics you should be familiar with:

### Accuracy 
### precision 
### Recall
### F1-score
### Confusion matrix 
### Classification report(from sckit-learn) - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

"""

# check the accuracy of our model
loss, accuracy = model_10.evaluate(x_test, y_test)
print(f"Model loss on the test set:{loss}")
print(f"Model accuracy on the test set:{(accuracy * 100):.2f}%")

"""## How about a confusion matrix? """

# Create a confusion matrix

from sklearn.metrics import confusion_matrix

# make predictions

y_preds = model_10.predict(x_test)

# Create cofusion matrix

confusion_matrix(y_test, y_preds)

y_test[:10]

y_preds[:10]

"""### Oops... looks like our predictions array has come out in "prediction probality" from ... the standard output from the sigmoid (or softmax) activation functions."""

# convert prediction probabilities to binary format and view the first 10

tf.round(y_preds)[:10]


# Create a confusion matrix
confusion_matrix(y_test, tf.round(y_preds))

"""## How about we prettify our confusion matrix?

"""

# the confusion matrix code we're about to write is a remix of the scikit-learn's plot_confusion_matrix
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# and Made with ML's introductory notebook - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb


import itertools

figsize = (10, 10)

# Create the confusion matrix
cm = confusion_matrix(y_test, tf.round(y_preds))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize our confusion matrix
n_classes = cm.shape[0]

# Let's prettify it
fig, ax = plt.subplots(figsize=figsize)
# Create a matrix plot
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)
classes = False

if classes:
    labels = classes
else:
    labels = np.arange(cm.shape[0])
# Label the axes
ax.set(title="confusion Matrix",
       xlabel="predicted Label",
       ylabel="True Label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       xticklabels=labels,
       yticklabels=labels)
# Set x-axis labels to bottom
ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()
# Adjust label size
ax.yaxis.label.set_size(20)
ax.xaxis.label.set_size(20)
ax.title.set_size(20)

# Set threshold for different colors
threshold = (cm.max() + cm.min()) / 2.
# plot the text on each cell
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=15)

cm.shape[0]

"""# Working with a larger example (multiclass classification)

### When you have more than two classes as an option, it's known as "multi-class classification".
### This means if you have 3 different classes, it's multi-class classification.
### It also means if you have 100 different classes,it's multi-class classification.

### To practice multi-class classification, we're going to build a neural network to classify images of different items of clothing.
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# the data has already been sorted into training and test sets for us
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# show the first training example
print(f"Training sample:\n{train_data[0]}\n")
print(f"Training label:\n{train_data[0]}\n")

# check the shape of a single example
train_data[0].shape, train_labels[0].shape

# plot a single sample
import matplotlib.pyplot as plt

plt.imshow(train_data[7])

# check out samples label
train_labels[7]

# Create a small list so we can index onto our training labels so they're human-readable

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","shirt","sneaker","Bag","Ankle boot"]
len(class_names)

# plot an example image and its label
index_of_choice = 2000
plt.imshow(train_data[index_of_choice],cmap=plt.cm.binary)
plt.title(class_names[train_labels[index_of_choice]])

# plot multiple random images of fashion MNIST
import random
plt.figure(figsize= (7,7))
for i in range(4):
  ax = plt.subplot(2,2,i+1)
  rand_index = random.choice(range(len(train_data)))
  plt.imshow(train_data[rand_index], cmap= plt.cm.binary)
  plt.title(class_names[train_labels[rand_index]])
  plt.axis(False)

"""##Building a multi-class classification model 

### For our multi-class classification model, we can use a similar architecture to our binary classifiers, however, we're going to have to tweak a few things:

#### Input shape = 28*28(the shape of one image)
#### Output shape = 10 (one per class of clothing)
#### Loss function = tf.keras.losses.CategoricalCrosstentropy()
#### Output layer activation = Softmax (not sigmoid)
"""

# set random seed
tf.random.set_seed(42)

# Create the model
model_11 = tf.keras.Sequential