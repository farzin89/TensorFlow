# -*- coding: utf-8 -*-
"""04_transfer_learning_in_tensorflow_part_1_feature_extraction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-CP75SJhYZchvVTLs2LGRVAQdLzo0lK8

### Transfer Learning with Tensorflow Part 1 : Feature Extraction 

#### Transfer learning is leveraging a working model's existing architecture and learend patterns for our own problem. There two main benefits:

##### 1. Can leverage an existing neural network architecture proven to work on problems similar to our own.
##### 2. Can leverage a working neural network architecture which has alredy learned patterns on similar data to our own, then we can adapt those patterns to our own data
"""

# are we using a GPU?
!nvidia-smi

## Dowloading and becoming one with the data
import zipfile

# Download the data 
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip
# Unzip the downloaded file 
zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip")
zip_ref.extractall()
zip_ref.close()

# how many images in each folder?

import os 

# Walk through 10 percent data directory and list number of files
for dirpath,dirnames,filenames,in os.walk("10_food_classes_10_percent"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

"""## Creating data loaders (preparing the data)

### we'll use the ImageGataGenerator class to load in our images in bathes.


"""

# Setup data inputs 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SHAPE = (224,224)
BATCH_SIZE = 32

train_dir= "10_food_classes_10_percent/train/"
test_dir =  "10_food_classes_10_percent/test/"

terain_datagen = ImageDataGenerator(rescale = 1/255.)
test_datagen = ImageDataGenerator(rescale = 1/255.)

print("Training images:")
train_data_10_percent = terain_datagen.flow_from_directory(train_dir,
                                                           target_size = IMAGE_SHAPE,
                                                           batch_size = BATCH_SIZE,
                                                           class_mode = "categorical")

print("Testing images:")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size = IMAGE_SHAPE,
                                             batch_size =BATCH_SIZE,
                                             class_mode ="categorical")

""" ## Setting up callbacks(things to run whilst our model trains)

Callbacks are extra functionality you can add to your models to be performed during or after training. Some of the most popular callbacks:

* Tracking  experiments with the the TensorBoard callback 
* Model checkpoint with the ModelChechpoint callback
* Stopping a model from training (before it trains too long and overfits) with the EarlyStopping callback
"""

# Create TensorBoard callback (functionized because we need to create a new one for each model )

 import datetime 

 def create_tensorboard_callback (dir_name,experiment_name):
   log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
   print(f"saving TensorBoard log files to : {log_dir}")
   return tensorboard_callback

"""** Note**: you can customize the directory where you TensorBoard logs(model training metrics) get saved to whatever you like. The log_dir parameter we've created above is only one option.

## Creating models using TensorFlow Hub

In the past we've used TensorFlow to create our own models layers by layer from scratch.

Now we're going to do a similar process, except the majority of our model's layers are going to come from TensorFlow Hub.

We can access pretrained models on : https://tfhub.dev/

Browsing the Tensorflow Hub page and sorting for image classification, we found the following feature vector model link : https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1
"""

# Let's compare the following two models
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

# import depebdencies

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

IMAGE_SHAPE+(3,)

# Let's make a create_model() function to create a model from a URL

def create_model (model_url,num_classes=10):
  """
  Takes a TensorFlow Hub URL and creates a keras sequential model with it.

  Args:
    model_url(str): A TensorFlow Hub feature extraction URL.
    num_classes (int) : Number of output neurons in the output layer,
     should be equal to number of target classes,default 10.
  Return:
     An uncompliled Keras Sequentioal model with model_url as feature extractor
     layer and Dense output layer with num_classes output neurons.
  """

  # download the pretrained model and save it as a keras layer
  feature_extractor_layer  = hub.KerasLayer(model_url,
                                            trainable = False, # freeze the already learned patterns
                                            name = "feature_extraction_layer",
                                            input_shape = IMAGE_SHAPE+(3,))
  # Create our own model

  model = tf.keras.Sequential([
      feature_extractor_layer,
      layers.Dense(num_classes,activation="softmax",name="output_layer")
  ])


  return model

"""### Creating and Testing ResNet TensorFlow hub Feature Extraction model """

# Create Resent model

resnet_model = create_model(resnet_url,
                            num_classes = train_data_10_percent.num_classes)

# Compile our resnet model
resnet_model.compile(loss = "categorical_crossentropy",
                     optimizer = tf.keras.optimizers.Adam(),
                     metrics = ["accuracy"])

