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



