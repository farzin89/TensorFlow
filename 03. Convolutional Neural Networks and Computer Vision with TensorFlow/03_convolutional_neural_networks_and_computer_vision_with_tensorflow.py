# -*- coding: utf-8 -*-
"""03. Convolutional Neural Networks and Computer Vision with TensorFlow.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1If7ZGs8Hv6Y6-uLofCXAYtIGzKoK4BdS

## Introduction to Convolution Neural Networks and Computer Vision with TensorFlow

### computer vision is the practice of writing algorithms which can discover patterns in vissual . Such as the camera of a self - driving car recognizing the car in front.

## Get the data 

### The images we're working with are from the Food101 dataset (101 different classes of food): https://www.kaggle.com/dansbecker/food-101

However we've modified it to only use two classes (pizza & steak) using the image data modification notebook: https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/image_data_modification.ipynb

## Note : We start with a smaller dataset so we can experiment quickly and figure what works (or better yet what doesn't work) before scaling up.
"""

import zipfile

!wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

# Unzip the downloaded file 

zip_ref = zipfile.ZipFile("pizza_steak.zip")
zip_ref.extractall()
zip_ref.close

"""## Inspect the data (become one with it)

#### A very crucial step at the beginning of any machine learning project is becoming one with the data.

And for a computer vision project... this usually means visualizing many samoles of your data.
"""

!ls pizza_steak

!ls pizza_steak/train/

!ls pizza_steak/train/steak

import os 

# Walk through pizza_steak directory and list number of files 

for dirpath,dirnames,filenames in os.walk("pizza_steak"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# The extra file in our pizza_steak directory is ".DS_store"
!ls -la pizza_steak

# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))
num_steak_images_train

"""### To visualize our images, first Let's get the class names programmatically."""

# Get the classsnames programmatically 

import pathlib 
import numpy as np
data_dir = pathlib.Path("pizza_steak/train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*") ])) # Created a list of class_names from the subdirectories
print(class_names)

# Let's visualize our images 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import random 

def view_random_image(target_dir,target_class):
  # Setup the target directory (we'll view images from here)
  target_folder = target_dir+target_class
 
  # Get a random image path 
  random_image = random.sample(os.listdir(target_folder),1)
  print(random_image)
  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image


return img

# View a random image from the training dataset

img = view_random_image(target_dir="pizza_steak/train/",
                        target_class="pizza")

# The images we've imported and plotted are actually gaint arrays/tensors of different pixel values

import tensorflow as tf

tf.constant(img)

# View the image shape
img.shape  # Returns width, height, colour channels

"""## Note : As we've discussed before, many machine learning models, including neural networks prefer the values they work with to be between 0 and 1. knowing this , one of the most common preprocessing steps for working with images is to scale (also referred to as normalize) their pixel values by dividing the image array by 255.(since 255 is the maximum pixel value)."""

# Get all the pixel values between 0 & 1

img / 255

"""## An end-to-end example

### Let's build a convolutional neural network to find patterns in our images, more specifically we a need way t:  

#### - Load our images 
#### - Preprocess our images 
#### - Build a CNN to find patterns in our images 
#### - Compile our CNN 
#### - Fit the CNN to our training data 



"""

from tensorflow.python.keras.backend import binary_crossentropy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# set the seed

tf.random.set_seed(42)

# Preprocess data (get all of pixel values between 0 & 1 , also called scaling/normalization)

train_datagen = ImageDataGenerator(rescale=1. / 255)
valid_datagen = ImageDataGenerator(rescale=1. / 255)

# Setup paths to our data directories
train_dir = "/content/pizza_steak/train"
test_dir = "pizza_steak/test"

# Import data from directories and turn it into batches

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)
valid_data = train_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

# Build a CNN model (same as the tiny VGG on the CNN explainer Website)

model_1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=10,
                         kernel_size=3,
                         activation="relu",
                         input_shape=(224, 224, 3)),

  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2,
                            padding="valid"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid")

])

# Compile our CNN
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"]
                )
# fit the model

history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

"""### Note : If the above cell is talking longer than 10 seconds per epoch, make sure you're using a GPU by going to Runtime-> Change Runtime Type-> Hardware Acccelator -> GPU(you may have to rerun some cell above)"""

# Get a model summary

model_1.summary()

"""
### Practice/exercise: Go through the CNN explainer website for a minimum of 10-minute and compare our neural network with thiers : https://poloclub.github.io/cnn-explainer/"""

## Using the same model as before

### Let's replicate the model we've built in a previous section to see if it works with our image data.

### the model vwe're building is from the [Tensorflow playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.69596&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false).
"""

# Set Random seed 

tf.random.set_seed(42)

# Creeate a model to replicate the Tensorflow playground model 

model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224,224,3)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

# compile the model 

model_2.compile(loss="binary_crossentropy",
                optimizer= tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])
# Fit the model
history_2= model_2.fit(train_data,
                       epochs=5,
                       steps_per_epoch= len(train_data),
                       validation_data= valid_data,
                       validation_steps = len(valid_data))

# Get a summary of model_2
model_2.summary()

"""### Despite having 20x more parameters than our CNN (model_1),model_2 performs terribly... let's try to improve it.

"""

# set the random seed
tf.random.set_seed(42)

# create the model (same as above let's step it up notch)
model_3 = tf.keras.Sequential([
   tf.keras.layers.Flatten(input_shape=(224,224,3)),
   tf.keras.layers.Dense(100,activation="relu"),
   tf.keras.layers.Dense(100,activation="relu"),
   tf.keras.layers.Dense(100,activation="relu"),
   tf.keras.layers.Dense(1,activation="sigmoid")
])

# Compile the model 
model_3.compile(loss= "binary_crossentropy",
                optimizer =tf.keras.optimizers.Adam(),
                metrics =["accuracy"])
# Fit the model 
history_3= model_3.fit(train_data,
                       epochs=5,
                       steps_per_epoch= len(train_data),
                       validation_data= valid_data,
                       validation_steps = len(valid_data))

# Get the summary of model_3
model_3.summary()

"""##Note:
####you can think of trainable parameters as pattern a model can learn from data. intuitively, you might think more is better. and in lots of cases, it is. But in this case, the difference here is the two different styles of model we're using. where a series of dense layers has a number of different learnable parameters connected to each other and hence a higher number of possible learning patterns, a convolutional neural network seeks to sort out and learn the most important patterns in an image. So even though these less learnable parameters in our convolutional neural network, these are often more helpful in dechiphering between different features in an image.
"""

model_1.summary()

"""### Binary Classification: Let's break it down

#### 1. Become one with the data(visualize,visualize,visualize)
#### 2. Preprocess the data (prepared it for our model,the main step here was scaling/normalizing)
#### 3. Created a model (start with a baseline)
#### 4. fit the model
#### 5. Evaluate the model
#### 6. Adjust different parameters and improve the model (try to beat our baseline)
#### 7. Repeat until satisfied (experiment,experiment,experiment)

## 1. Become one with data
"""

# Visualize data
plt.figure()
plt.subplot(1,2,1)
steak_img= view_random_image("pizza_steak/train/","steak")
plt.subplot(1,2,2)
pizza_img = view_random_image("pizza_steak/train/","pizza")

"""### 2. Preprocess the data (prepare it for a model)



# Define directory dataset paths 
train_dir= "pizza_steak/train/"
test_dir = "pizza_steak/test/"

### Our next step is to turn our date into "batches"

#### A Batch is a small subset of data. Rather than look at all 10,000 images, a model might only look at 32 at a time.

#### It does this for a couple of reasons:
##### 1. 10,000 images(or more) might not fit into the memory of your processor(GPU)
#### 2. Trying to learn the patterns in 10,000 images in one hit could result in the model not being able to learn very well

### why 32?
#### Because 32 is good for your health ...


# Create train and test generators and rescale the data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

# Load in our image data from directories and turn them into batches

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(224,224),
                                               class_mode="binary",
                                               bath_size=32)
test_data = test_datagen.flow_from_directory(directory=train_dir,
                                             target_size=(224,224),
                                             class_mode= "binary",
                                             batch_size=32)

# Get a sample of a train data batch
images,labels = train_data.next() # get the "next" batch of images/labels in train_data
len(images),len(labels)

# How many batches are there?
len(train_data)

images[:2], images[0].shape

# View the first batch of labels
labels

"""### 3. create a CNN model (start with a baseline)

A baseline is a relatively simple model or existing result that you setup when beginning a machine learning experiment and then as you keep experimenting, you try to beat the baseline.

## Note : In deep learning , there is almost an infinite amount of architectures you could create. So one the best ways to get started is  to start with something simple and see if it works on your data and then introduce complexity as required(e.g look at which current model is performing best in the field for your problem).
"""

# Make the creating of our model a little easier

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Activation
from tensorflow.keras import Sequential

# Create the model(this will be our baseline,a layer convolutional neural network )
model_4 = Sequential([
     Conv2D(filters=10, # filter is the number of sliding windows going across an input(higher=more complex model)
            kernel_size=(3,3), # the size of the sliding window going across an input
            strides=(1,1), # the size of the step the sliding window takes across an input # by difult is 1
            padding="valid", # by difult is valid # if "same",output is same as input shape,if "valid",output shape gets compressed
            activation="relu",
            input_shape = (224,224,3)), # input layer (specify input shape)
    Conv2D(10,3,activation="relu"),
    Conv2D(10,3,activation="relu"),
    Flatten(),
    Dense(1,activation="sigmoid") # output layer (working with binary classification so only 1 output neuron)
])

"""### practice: 
#### understand what's going on in a Conv2D layers by going through the CNN explainer website for 10-20 minutes: https://poloclub.github.io/cnn-explainer/#:~:text=CNN%20Explainer%20was%20created%20by%20Jay%20Wang%20%2C,research%20collaboration%20between%20Georgia%20Tech%20and%20Oregon%20State.
"""

from tensorflow.keras import optimizers
# compile the model
model_4.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

"""### Fit the model 

"""

# Check the lengths of training and test data generators
len(train_data), len(test_data)

# Fit the model
history_4= model_4.fit(train_data, # this is a combination of labels and sample data
                       epochs=5,
                       steps_per_epoch=len(train_data),
                       validation_data= test_data,
                       validation_steps=len(test_data))

model_1.evaluate(test_data)

model_1.summary()

"""### 5. Evaluating our model 

#### it looks like our model is learning something, let's evaluate it.
"""

import pandas as pd
pd.DataFrame(history_4.history).plot(figsize=(10,7))
from tensorflow.python.keras.metrics import accuracy
# Plot the validation and training curves separately
def plot_loss_curves(history):
   """
   Returns separate loss curves for training and validation metrics."""
   loss = history.history["loss"]
   val_loss= history.history["val_loss"]

   accuracy= history.history["accuracy"]
   val_accuracy = history.history["val_accuracy"]

   epochs= range(len(history.history["loss"])) # how many epochs did we run for?

   # Plot loss
   plt.plot(epochs,loss,label="training_loss")
   plt.plot(epochs,val_loss, label="val_loss")
   plt.title("loss")
   plt.xlabel("epochs")
   plt.legend()

  # plot the accuracy
   plt.figure()
   plt.plot(epochs,accuracy,label="training_accuracy")
   plt.plot(epochs,val_accuracy, label="val_accuracy")
   plt.title("accuracy")
   plt.xlabel("epochs")
   plt.legend()

"""### Note:
#### When a model's validation loss starts to increase ,it's likely that the model is overfitting the training dataset. This means, it's learning the patterns in the traning dataset too well and thus the model's ability to generalize to unseen data will be diminished.
"""

# Chech out the loss and accuracy of model_4
plot_loss_curves(history_4)

"""### Note:

#### idealy the two loss curves(training and validation) will be very similar to each other (training loss and validation loss decreasing at similar rates),when there are large differences your model may be overfiting.

### 6. Adjust the model parameters 
#### Fiting a machine learning model comes in 3 steps:

##### 0. Create a baseline 
##### 1. Beat the baseline by overfitting a larger model
##### 2. Reduce overfiting 

### ways to induce overfitting:

#### * Increase the number of conv layers
#### * Increase the number of conv filters 
#### * Add another dense layers to the output of our flattened layer

### Reduce overfitting:

#### * Add data augmentation
#### * Add regularization layers(such as Maxpool2D)
#### * Add more data...

### Note :   

#### Reducing overfitting is also known as regularization.
"""

# Create the model (this is going to be our new baseline)

model_5 = Sequential([
     Conv2D(10,3,activation="relu",input_shape = (224,224,3)),
     MaxPool2D(pool_size=2),
     Conv2D(10,3,activation="relu"),
     MaxPool2D(),
     Conv2D(10,3,activation="relu"),
     MaxPool2D(),
     Flatten(),
     Dense(1,activation="sigmoid")
])

# Compile the model
model_5.compile(loss= "binary_crossentropy",
                optimizer = Adam(),
                metrics = ["accuracy"])

# Fit the model
history_5 = model_5.fit(train_data,
                        epochs= 5,
                        steps_per_epoch = len(train_data),
                        validation_data = test_data,
                        validation_steps= len(valid_data))

# Get a summary of our model with max pooling
model_5.summary()

model_4.summary()

# plot loss curves
plot_loss_curves(history_5)

"""### opening our bag of tricks and finding data augmentation"""

from keras_preprocessing.image import image_data_generator
# Create ImageDataGenerator training instance with data augmentation
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.3,
                                             horizontal_flip=True)
# Create ImageGenerator without data augmentation
train_datagen = ImageDataGenerator(rescale=1/255.)

# Create ImageDataGenerator without data augmentation for the test dataset
test_datagen = ImageDataGenerator(rescale=1/255.)

"""### Question : 

#### what is data augmentation?
#### Data augmentation is the process of altering our traning data,leading it to have more diversity and in turn allowing our models to learn more generalizable(hopefully) patterns. Altering might mean adjusting the rotation of an image,flipping it,cropping it or something similar.

### Let's write some code to visualize data augmentation...
"""

# import data and augment it from training directory
print("Augmented training data")
train_data_augmented = train_datagen_augmented.flow_from_directory(train_dir,
                                                                   target_size=(224,224),
                                                                   batch_size=32,
                                                                   class_mode="binary",
                                                                   shuffle=False) # for demonstration purposes only

# Create non_augmented train data batches
print("Non-augmented training data:")
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224,224),
                                               batch_size=3,
                                               class_mode="binary",
                                               shuffle=False)
IMG_SIZE =(224,224)
# Create non-augmentet test data batches
print("Non_augmented test data:")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=IMG_SIZE,
                                             batch_size=32,
                                             class_mode="binary")

"""###Note:
#### Data augmentation is usually only performed on the training data.using 'ImageDataGenerator' built-in data augmentation parameters our images are left as they are in the directories but are modigied as they're loaded into the model.

#### Finaly let's visualize some augmented data!!!

"""

# Get sample data batches

images,labels = train_data.next()
augmented_images, augmented_labes= train_data_augmented.next() #note: label aren't augmented... only data(images)

# show original image and augmented image
import random
random_number = random.randint(0,32) # our batch size are 32...
print(f"showing image number:{random_number}")
plt.imshow(images[random_number])
plt.title(f"Original image")
plt.axis(False)
plt.figure()
plt.imshow(augmented_images[random_number])
plt.title(f"Augmented image")
plt.axis(False)

"""#### Now we've seen what augmented training data looks like,let's build a model and see how it learns on augmented data.

"""

# Create a model (same as model_5)
model_6 =Sequential([
    Conv2D(10,3,activation="relu"),
    MaxPool2D(pool_size=2),
    Conv2D(10,3,activation="relu"),
    MaxPool2D(),
    Conv2D(10,3,activation= "relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1,activation="sigmoid")
])

# compile the model

model_6.compile(loss= "binary_crossentropy",
                optimizer = Adam(),
                metrics = ["accuracy"])

# Fit the model
history_6 = model_6.fit(train_data_augmented, # fitting model_6 on augmented training data
                        epochs = 5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data= test_data,
                        validation_steps=len(test_data))

plot_loss_curves(history_6)

"""###3 Let's shuffle our augmented training data and train another model (the same as before) on it and see what happens."""

# Import data and augment it and shuffle from training directory
train_data_augmented_shuffled=train_datagen_augmented.flow_from_directory(train_dir,
                                                  target_size=(224,224),
                                                  class_mode="binary",
                                                  batch_size=32,
                                                  shuffle = True) # shuffle data this time

# Create the model (same as model_5 and model_6)
model_7 = Sequential([
      Conv2D(10,3,activation="relu",input_shape=(224,224,3)),
      MaxPool2D(),
      Conv2D(10,3,activation="relu"),
      MaxPool2D(),
      Conv2D(10,3,activation="relu"),
      MaxPool2D(),
      Flatten(),
      Dense( 1,activation="sigmoid")

])

# compile the model
model_7.compile(loss= "binary_crossentropy",
                optimizer =Adam(),
                metrics= ["acccuracy"])
# fit the model
history_7 = model_7.fit(train_data_augmented_shuffled, # we're fitting on augmented and shuffled data now
                        epochs = 5,
                        steps_per_epoch=len(train_data_augmented_shuffled),
                        validation_data= test_data,
                        validation_steps=len(test_data))

# plot the lost curves
plot_loss_curves(history_7)

"""### Note:
#### when shuffling training data,the model gets exposed to all different kinds of data during training,thus enabling it to learn features across a wide array of images(in our case,pizza & steak at the same time instead of just pizza then steak)

### 7. Repeat until satisfied

#### Since we've already beaten our baseline,there are a few things we could try to contine to improve our model:

#### * Increase the number of model layers (e.g. add more 'Conv2D'/MaxPool2D' layers)

#### * Increase the number of filters in each convolutional layer (e.g.from 10 to 32 or even 64)

#### * Train for longer (more epochs)
#### * Find an ideal learning rate 
#### * Get more data (give the model more opportunities to learn)
#### * use " Transfer learning" to leverage what another image model has learn and adjust it for our own use case 

### Practice : Recreate the model on the CNN explainer website(same as model_1) and see how it performs on the augmented shuffled training data.

### Making a prediction with our trained on our own custom data
"""

# Classes we"re working with
print(class_names)

# view our example image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg
steak =mpimg.imread("03-steak.jpeg")
plt.imshow(steak)
plt.axis(False)

# check the shape of our image
steak.shape

"""### Note: 
#### when you a train a neural network and you want to make a prediction with it your own custom data, it's important than your custom data(or new data) is preprocessed into the same format as the data your model was trained on.

"""

# create a function to import an image and resize it to be able to used with our model

def load_and_prep_image(filename,img_shape=224):
  """
  Reads an image from filename, turns it into a tensor and reshapes it
   to (img_shape,img_shape,colour_channels).
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode the read file into a tensor
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img,size=[img_shape,img_shape])
  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

# Load in and preprocess our custom image
steak = load_and_prep_image("03-steak.jpeg")
steak

model_7.predict(tf.expand_dims(steak, axis = 0))
pred

"""### Looks like our custom image is being put through our model,however, it currently outputs a prediction probability , wouldn't it be nice if we could visualize the image as well as the model's prediction?

"""

# Remind ourselves of our class names
class_names

# We can index the predicted class by rounding the prediction probability and indexing it on the class names.
pred_class = class_names[int(tf.round(pred))]
pred_class

def pred_and_plot(model,filename,class_names=class_names):
   """
   Import an image located at filename ,make a prediction with model and
   plot the image with the predicted class as the title.
   """
   # Import the target image and preprocess it
   img= load_and_prep_image(filename)

   # make a prediction
   pred = model.predict(tf.expand_dims(img,axis=0))

   # Get the predicted class
   pred_class = class_names[int(tf.round(pred))]

   # plot the image and predicted class
   plt.imshow(img)
   plt.title(f"prediction: {pred_class}")
   plt.axis(False)

# Test our model on a custom image
pred_and_plot(model_7,"03-steak.jpeg")

"""### our model works! Let's try it on another image... this time pizza 


"""

# download another test custom image and make a prediction on it
!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-pizza-dad.jpeg

pred_and_plot(model_7,"03-pizza-dad.jpeg")

"""### Multi-class image Classification 
#### We've just been through a bunch of the following steps with a binary classification problem (pizza vs. steak), now we're going to step things up a notch with 10 classes of food (multi-class classification).

#### 1. Become one with the data
#### 2. Preprocess the data (get it ready for a model)
#### 3. Create a model (start with a baseline)
#### 4. Fit the model (overfit it to make sure it works)
#### 5. Evaluate the model 
#### 6. Adjust different hyperparameters and improve the model (try to beat baseline/reduce overfitting)
#### 7. Repeat until satisfied

## 1. Import and become one with data
"""

import zipfile
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip

# Unzip our data
zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip","r")
zip_ref.extractall()
zip_ref.close()

import os

# Walk through 10 classes of food image data
for dirpath,dirnames,filenames in os.walk("10_food_classes_all_data"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

!ls -la 10_food_classes_all_data/

# Setup the train and test directories

train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"

# let's get the class names
import pathlib
import numpy as np
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)

# Visualize , visualize , visualize

import random
img = view_random_image(target_dir= train_dir,
                        target_class = random.choice(class_names))

"""## 2. Preproces the data (prepare it for a model)"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rescale
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator (rescale=1/255.)

# Load data in from directories and turn it into batches

train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size =(224,244),
                                               batch_size =32,
                                               class_mode ="categorical")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(244,244),
                                             batch_size = 32,
                                             class_mode ="categorical")

"""### Create a model (start with a baseline)

#### we've been talking a lot about the CNN explainer website... how about we just take their model (also on 10 classes) and use it for our problems..?
"""



