# -*- coding: utf-8 -*-
"""07_food_vision_milestone_project_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G3D42hXIHY2pfxh3LAIuHcKapb6cz98c

## Milestone Project 1 : Food Vision Big

### Check GPU

Google Colab offers free GPUs (thank you google), however, not all of them are compatible with mixed precision training.PendingDeprecationWarning.

Google Colab offers: 
* K80(not compatible)
* P100 (not compatible)
* Tesla T4 (compatible)

Knowing this, in order to use mixed precison training we need access to a Tesla T4(from within Google Colab) or if we're using our own hardware, our GPU needs a score of 7.0+ (see here: https://developer.nvidia.com/cuda-gpus)
"""

!nvidia-smi -L


"""## Get helper function 

In past modules, we've created a bunch of helper functions to do small tasks required for our notebooks.

rather than rewrite all of these, we can import a script and load them in from there.

The script we've got available can be found on GitHub : https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py


"""

# download helper function script

!wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py

# Import series of helper functions for the notebook
from helper_functions import create_tensorboard_callback,plot_loss_curves,compare_historys

"""## Use TensorFlow Datasets to download Data 

If you want to get an overview of Tensorflow Datasets (TFDS), read the guide : 
https://www.tensorflow.org/datasets/overview

"""

# Get TensorFlow Datasets
import tensorflow_datasets as tfds

# List all available datasets
datasets_list = tfds.list_builders() # get all avaiable datasets in TFDS
print("food101" in datasets_list) # is our target dataset in the list of TFDS dataset?

# Load in the data (takes 5-6 minutes in Google Colab)
(train_data,test_data),ds_info = tfds.load(name="food101",
                                           split=["train","validation"],
                                           shuffle_files = True,
                                           as_supervised = True, # data gets returned in tuple format(data,label)
                                           with_info = True)
