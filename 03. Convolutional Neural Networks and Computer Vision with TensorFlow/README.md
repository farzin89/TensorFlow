## Introduction to Convolution Neural Networks and Computer Vision with TensorFlow

### computer vision is the practice of writing algorithms which can discover patterns in vissual . Such as the camera of a self - driving car recognizing the car in front.

## Get the data 

### The images we're working with are from the Food101 dataset (101 different classes of food): https://www.kaggle.com/dansbecker/food-101

However we've modified it to only use two classes (pizza & steak)
 using the image data modification notebook: https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/image_data_modification.ipynb

## Note : We start with a smaller dataset so we can experiment quickly and figure what works (or better yet what doesn't work) before scaling up.


## Inspect the data (become one with it)

#### A very crucial step at the beginning of any machine learning project is becoming one with the data.

And for a computer vision project... this usually means visualizing many samoles of your data.

## Note : As we've discussed before, many machine learning models,
 including neural networks prefer the values they work with to be between
 0 and 1. knowing this , one of the most common preprocessing steps for
 working with images is to scale (also referred to as normalize) their pixel
 values by dividing the image array by 255.(since 255 is the maximum pixel value).

## An end-to-end example

### Let's build a convolutional neural network to find patterns in our images, more specifically we a need way t:  

#### - Load our images 
#### - Preprocess our images 
#### - Build a CNN to find patterns in our images 
#### - Compile our CNN 
#### - Fit the CNN to our training data 

### Practice/exercise: Go through the CNN explainer website for 
a minimum of 10-minute and compare our neural network with thiers : https://poloclub.github.io/cnn-explainer/

### the model vwe're building is from the [Tensorflow playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.69596&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false).

##Note:
####you can think of trainable parameters as pattern a model can learn
 from data. intuitively, you might think more is better. and in lots of cases, it is.
 But in this case, the difference here is the two different styles of model we're
 using. where a series of dense layers has a number of different learnable parameters 
connected to each other and hence a higher number of possible learning patterns,
 a convolutional neural network seeks to sort out and learn the most important patterns 
in an image. So even though these less learnable parameters in our convolutional neural 
network, these are often more helpful in dechiphering between different features in an image.

### Binary Classification: Let's break it down

#### 1. Become one with the data(visualize,visualize,visualize)
#### 2. Preprocess the data (prepared it for our model,the main step here was scaling/normalizing and turning our data into batches)
#### 3. Created a model (start with a baseline)
#### 4. fit the model 
#### 5. Evaluate the model  
#### 6. Adjust different parameters and improve the model (try to beat our baseline)
#### 7. Repeat until satisfied (experiment,experiment,experiment)

### Our next step is to turn our date into "batches"

#### A Batch is a small subset of data. Rather than look at all 10,000 images, a model might only look at 32 at a time.

#### It does this for a couple of reasons:
##### 1. 10,000 images(or more) might not fit into the memory of your processor(GPU)
#### 2. Trying to learn the patterns in 10,000 images in one hit could result in the model not being able to learn very well 

### why 32?
#### Because 32 is good for your health ...

### 3. create a CNN model (start with a baseline)

A baseline is a relatively simple model or existing result that you setup when beginning
 a machine learning experiment and then as you keep experimenting, you try to beat the 
baseline.

## Note : In deep learning , there is almost an infinite amount of architectures you could 
create. So one the best ways to get started is  to start with something simple and see if 
it works on your data and then introduce complexity as required(e.g look at which current 
model is performing best in the field for your problem).

### practice: 
#### understand what's going on in a Conv2D layers by going through the CNN explainer website
 for 10-20 minutes: https://poloclub.github.io/cnn-explainer/#:~:text=CNN%20Explainer%20was%20created%20by%20Jay%20Wang%20%2C,research%20collaboration%20between%20Georgia%20Tech%20and%20Oregon%20State.

### Note:
#### When a model's validation loss starts to increase ,it's likely that the model is 
overfitting the training dataset. This means, it's learning the patterns in the traning 
dataset too well and thus the model's ability to generalize to unseen data will be diminished.

### Note:

#### idealy the two loss curves(training and validation) will be very similar to each other 
(training loss and validation loss decreasing at similar rates),when there are large differences your model may be overfiting.

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

### Question : 

#### what is data augmentation?
#### Data augmentation is the process of altering our traning data,
leading it to have more diversity and in turn allowing our models to
 learn more generalizable(hopefully) patterns. Altering might mean 
adjusting the rotation of an image,flipping it,cropping it or something similar.

###Note:
#### Data augmentation is usually only performed on the training data.
using 'ImageDataGenerator' built-in data augmentation parameters our images
 are left as they are in the directories but are modigied as they're loaded into the model.

#### Finaly let's visualize some augmented data!!!

### Let's write some code to visualize data augmentation...

#### Now we've seen what augmented training data looks like,let's build a model and see
 how it learns on augmented data.

###3 Let's shuffle our augmented training data and train another model 
(the same as before) on it and see what happens.

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

### Note: 
#### when you a train a neural network and you want to make a prediction with
 it your own custom data, it's important than your custom data(or new data) is preprocessed into the same format as the data your model was trained on.


### Multi-class image Classification 
#### We've just been through a bunch of the following steps with a binary classification
 problem (pizza vs. steak), now we're going to step things up a notch with 10 classes of 
food (multi-class classification).

#### 1. Become one with the data
#### 2. Preprocess the data (get it ready for a model)
#### 3. Create a model (start with a baseline)
#### 4. Fit the model (overfit it to make sure it works)
#### 5. Evaluate the model 
#### 6. Adjust different hyperparameters and improve the model (try to beat baseline/reduce overfitting)
#### 7. Repeat until satisfied 






