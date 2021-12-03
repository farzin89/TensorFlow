# TensorFlow

## 3 steps in modelling with TensorFlow
1. **Creating a model** - define the input and output layers , as well as the hidden layers of a deep learning model.
2. **Compiling a model** - define the loss function (in others words,the function which tells our model how wrong it is) and the optimizer (tells our model how to improve the patterns its learning ) and evaluation metrics(what we can use to interpret the performance of our model).
3. **Fitting the a model** - learning the model try to find patterns between x & y (features and labels). 


## impriving our model

we can imporve our model,by altering the steps we took to create a model.

1. **Creating a model** - here we might add more layers,increase the number of hidden units (all called neurons) within each of the hideen layers,change the activation function of each layer.

2. **Compilig a model** - here we might change the optimization function or perhaps the **learning rate** of the optimization function.

3. **Fiting a model** - here we might fit a model for more **epochs**(give the more  more examples to learn from).

# common ways to improve a deep model:
1. adding layers 
2. increase the number of hidden units
3. change the activation functions
4. change the optimization function
5. change the learning rate 
6. fitting on more data
7. fitting for longer 

## Evaluating a model 

In practice , a typical workflow you will go through when building neural networks is :

'''
Build amodel -> fit it -> evaluate it-> tweak a model -> fit it -> evaluate it ->
tweak a model -> fit it -> evaluate it ...
'''
 when it comes to evaluation... there are 3 words you should memorize:
 "Visualize, visualize,visualize"

 its a good idea to visualize:
 * The data - what data are we working with ? what does it look like?  
 * The model itself - what does our model look like ?
 * The training of a model - how does a model perform while it learns? 
 * The predictions of the model - how do the predictions of a model line up aqainst the qround truth( the original labels)  

### the 3 sets ...

* **Training set** - the model learns from this data, which is typically 70 - 80 % of the total data you have available .
* **Validation set** - the model gets tuned on this data, which is typically 10-15% of the data avaiable .
* **Test set** - the model gets evaluated on this data to test what is has learned, this set is typically 10-15% of the total data avaiable .

(for example 1. course material = training set and 2. practice exam =validation set and 3 . final exam (test set)

## Evaluating our model's predictions with regression evaluation metrics

Depending on the problem you are working on,there will be different evaluationvmetrics to evaluate your model's performance.

since we are working on a regression , two of the mainn metrics :    

* MAE - mean absolute error,"on average, how wrong is each of my model's predictions"
* MSE - mean square error, " square the average errors"

### Running experiments to improve our model 

# Build a model -> fit it -> evaluate it -> tweak it -> fit it -> evaluate it -> tweak it -> fit it -> evaluate it ...

1. Get more data - get more examples for your model to train on (more opportunities to learn patterns or relationships between features and labels)

2. Make your model larger ( using a more complex model)- this might come in the form of more layers or more hidden units in each layer.

3. Train for longer - give your model more of a chance to find patterns in the data.

# lets do 3 modelling experiments:    

1. model 1 : - same as the original model, 1 layer, trained for 100 epochs.
2. model 2 : - 2 layers, trained for 100 epochs
3. model 3 : - 2 layers, trained for 500 epochs

#Note:
#### one of your main goals should be to minimize the time between your experiments. 
### the more experiments you do, the more things you will figure out which don't work and in turn, get closer to figuring out does work. Remember the machines learning practioner's motto: "experiment","experiment",experiment".

 ## Tracking your experiments

 #### one really good habit in machine learning modelling is to track the results of your experiments.

#### and when doing so, it can be tedious if you are running lots of experiments
### luckily, there are tools to help us.

## Resource: as you build more models, you will want to look into using :     
#### 1. TensorBoard -  a component of the Tensorflow library to help track modelling experiments 
#### 2 . Weights & Biases -  a tool for tracking all of kinds of machine learning experiments 


## saving our models

#### saving our models allows us to use them outside of google Colab(or wherever they were trained) such as in web application or a mobile app.

## there are two main formats we can save our model's too:
### 1. The SaveModel format
### 2. The HDF5 format 

## Download a model (or any other file) from google colab

#### if you want to download your files from Google Colab:     

#### 1 . you can go to the "files" tab and right click on the file you are after and click "download".

#### 2. Use code(see the cell below)

#### 3. save it to google drive by connecting Google Drive and copying it there (see 2nd code cell below)


### Note : when in our dataset we have number and letters we can use the one hot encoding 

 ### Let's try one- hot encode our DataFrame so it's all numbers
 pd.get_dummies(insurance)
 ((insurance is our dataset name))


### sklearn.model_selection.train_test_split https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
