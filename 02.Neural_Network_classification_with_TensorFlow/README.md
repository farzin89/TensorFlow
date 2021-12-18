## Itroduction to neural network classification with TensorFlow

#### in this notebook we're going to learn how to write neural networks for classification problems.

### A classification is where you try to classify something as one thing or another .

### A few type of classication problems:    

#### Binary classification 
#### Multiclass classification
#### Multiable classification 

## Steps in modelling  

### The steps in modelling with TensorFlow are typically:   

#### 1. Create or import a model 
#### 2. compile the model 
#### 3. Fit the model 
#### 4. Evaluate the model 
#### 5. Tweak 
#### 6. Evaluate

### since we're working on a binary on a classification problem and our model is getting around  50 % accuracy... it's performing as if it's gussing.

#### so lets things up a notch and add an extra layer.


## imrove our model

### let's look our bag of tricks to see how we can improve our model. 
 1. creat a model - we might to add more layers or increase the number of hidden units within a layer. 
 2. compiling a model - here we might to choose a diferent optimization function such as adam instead of SGD
 3. fiting a model - perhaps we might fit our model for more epochs (leave it training for longer).
  

#### To visulize our model's predictions,let's create a function 'plot_decision_boundary()' this function will :    
 ### - Take in a trained model,features(x) and labels(y)
 ### - Create a meshgred of the different x values
 ### - Make predictions across the meshgrid 
 ### - plot the predictions as well as a line between zones (where each unique class falls).

### Question : what's wrong the predictions we've made? Are we really evaluating our model correctly? hint: what data did the model learn on and what data did we predict on?

### Note : The combination of linear (straight lines) and non-linear(non-straight lines) functions is one of the key fundamentals of neural networks.

### Now we've discussed the concept of linear and non=linear functions (or lines), let's see them in sction.


 ## Evaluating and improving our classification

 #### so far we've been training and testing on the same dataset.. However,in machine learning this is basically a sin.

 #### let's create a teraining and test set.

"""### Note : For many problems, the loss function going down means the model is improving ( the prediction it's making are getting ground truth labels).
## Finfing the best learning rate 
 ### 1. A learning rate "Callback" - you can think of a callback as an extra piece of funtionality , you can add to your while it's training.
 ### 2. Another model (we could use the same one as above, but we're practicing building models here) 
 ### 3. a modified loss curves plot.

## More classification evaluation methods

### Alongside visualizing our models results as much as possible , there are a handful of other classification evaluation methods & metrics you should be familiar with:

### Accuracy 
### precision 
### Recall
### F1-score
### Confusion matrix 
### Classification report(from sckit-learn) - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

# Working with a larger example (multiclass classification)

### When you have more than two classes as an option, it's known as "multi-class classification".
### This means if you have 3 different classes, it's multi-class classification.
### It also means if you have 100 different classes,it's multi-class classification.

### To practice multi-class classification, we're going to build a neural network to classify images of different items of clothing.

##Building a multi-class classification model 

### For our multi-class classification model, we can use a similar architecture to our binary classifiers, however, we're going to have to tweak a few things:

#### Input shape = 28*28(the shape of one image)
#### Output shape = 10 (one per class of clothing)
#### Loss function = tf.keras.losses.CategoricalCrosstentropy()
#### Output layer activation = Softmax (not sigmoid)