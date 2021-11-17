# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rlCf0jbK9TTR41ntX2x-X8KSwBMi3J4q
"""

import tensorflow as tf
print(tf.__version__)

scalar = tf.constant(7)
scalar

scalar.ndim

#Creat a vector
vector = tf.constant([10,10])
vector

#check the dimension of our vector 
vector.ndim

#creat a matrix (has more than 1 dimention)
matrix = tf.constant([[10,7],
                      [7,10]])
matrix

matrix.ndim

#creat another matrix
another_matrix = tf.constant([[10.,7.],
                              [3.,2.],
                              [8.,9.]] , dtype = tf.float16) #specify the data type with dtype parameter
another_matrix

#what is the number dimentions of another_matrix?
another_matrix.ndim

#lets creat a tensor
tensor = tf.constant([[[1,2,3],
                       [4,5,6]],
                       [[7,8,9],
                        [10,11,12]],
                        [[13,14,15],
                        [16,17,18]]])
tensor

tensor.ndim

#what we have created so far:

# *scalar: a single number 
# *vector: a number with direction (e.g wind speed and direction)
# * matrix : a 2-dimentional array of numbers
#* Tensor : an n-dimentional array of numbers(whwn n can be any number,a 0-dimentional tensor is a scalar,a 1 dimentioal tensor is a vector)

#Creating tensors with tf.Variable 
tf.Variable

#Creat the same tensor with tf.Variable () as above

changeable_tensor = tf.Variable([10,7])
unchangeable_tensor = tf.constant([10,7])
changeable_tensor,unchangeable_tensor

#lets try to change one of the elements in our changeable tensor
changeable_tensor[0]= 7
changeable_tensor

# how about we try .assign()
changeable_tensor[0].assign(7)
changeable_tensor

# lets try change our unchanable tensor
unchangeable_tensor[0].assign(7)
unchangeable_tensor

# creating random trnsors

"""Crearing random tensors

random tensors are tensors of ability size which contain random numbers

"""



#ctreate two random(but the same) tensore
random_1 = tf.random.Generator.from_seed(7) #set seed for reproducibility
random_1 = random_1.normal(shape=(3,2))
random_2 = tf.random.Generator.from_seed(7)
random_2 = random_2.normal(shape=(3,2))
# are they equal?
random_1,random_2,random_1==random_2

"""###shuffle the order of elements in tensor"""

#shuffle a tensor (valuable for when you want to shuffle your data so the inherent order does not effect learning )
not_shufled = tf.constant([[10,7],
                           [3,4],
                           [2,5]])
#shuffle our non_shuffled tensor

tf.random.shuffle(not_shufled)

not_shufled

#shuffle our non_shuffled tensor

tf.random.shuffle(not_shufled)

#shuffle our non_shuffled tensor

tf.random.shuffle(not_shufled,seed=42)

"""::

**exercise:** Read through TensorFlow documentation on random seed generation:https://www.tensorflow.org/api_docs/python/tf/random/set_seed and practice writing 5 random tensors and shuffle them.

it looks like if we want our shuffled tensors to be in the same order, we have got to use the global level random seed as well as the operation level random seed

Rule 4 :"If both the global and the operation seed are set: Both seeds are used in conjunction to determine the random sequence.
"""

tf.random.set_seed(42) #global level random seed
tf.random.shuffle(not_shufled,seed=42) # operation level random seed

"""***other ways to make tensors

"""



# create a tensor of all ones
tf.ones([10,7])

#create a tensor of all zeroes
tf.zeros(shape=(3,4))

"""***Turn numpy array into tensors

the main difference between Numpy arrays and Tensorflow tensors is that tensors can be run on a GPU (much faster for numerical computing).

"""

# you can also turn numpy arrays into tensors
import numpy as np
numpy_A = np.arange(1, 25, dtype = np.int32) # create a Numpy array between 1 and 25
numpy_A 
# x = tf.constant(some_matrix) #capital for matrix or tensor
# y = tf.constant(vector) # non_caital for vector

A = tf.constant(numpy_A,shape = (2,3,4))
B = tf.constant(numpy_A)
A,B

A.ndim

"""###Geting information from tensor

when dealing with tensors you problably want to be aware of the following attributes:
* Shape
* Rank
* Axis or dimention
* Size

"""

# create a rannk 4 tensor (4 dimentions)
rank_4_tensor= tf.zeros(shape=[2,3,4,5])
rank_4_tensor

rank_4_tensor[0]

rank_4_tensor.shape,rank_4_tensor.ndim,tf.size(rank_4_tensor)

2*3*4*5

# Get variable attributes of our tensor

print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimentions(rank):", rank_4_tensor.ndim)
print("shape of tensore:",rank_4_tensor.shape)
print("Elements along the 0 axis:", rank_4_tensor.shape[0])
print("Elements along the last axis:",rank_4_tensor.shape[-1])
print("Total number of elements in our tensor:",tf.size(rank_4_tensor))
print("Total number of elements in our tensor:",tf.size(rank_4_tensor).numpy())

