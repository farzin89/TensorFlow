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
 * The data - what data are we wworking with ? what does it look like?  
 * The model itself - what does our model look like ?
 * The training of a model - how does a model perform while it learns? 
 * The predictions of the model - how do the predictions of a model line up aqainst the qround truth( the original labels)  
