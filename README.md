
<div style="text-align: right">INFO 7390 Advances in Data Sciences and Architecture </div>
<div style="text-align: right"> Suman Rawat : NUID 001058600</div>

# README : Assignment 3 (Deep Learning)

Convolutional neural networks are neural networks used primarily to classify images (i.e. name what they see), cluster images by similarity (photo search), and perform object recognition within scenes. For example, convolutional neural networks (ConvNets or CNNs) are used to identify faces, individuals, street signs, tumors, platypuses (platypi?) and many other aspects of visual data.

The efficacy of convolutional nets in image recognition is one of the main reasons why the world has woken up to the efficacy of deep learning. In a sense, CNNs are the reason why deep learning is famous. The success of a deep convolutional architecture called AlexNet in the 2012 ImageNet competition was the shot heard round the world. CNNs are powering major advances in computer vision (CV), which has obvious applications for self-driving cars, robotics, drones, security, medical diagnoses, and treatments for the visually impaired.

Create a GANs generator for some data (images, fonts, etc.)

#### Part A - Deep Learning model

In this Assignment we will apply a Deep Learning model to our data. Validate the accuracy on out of sample data.
The Deep Learning model that i have used is CNN(Convulation Neural Network). The purpose of this network is to detect the text of the CAPTCHA images.

 #### Part B - Activation function

The activation function is a node that is put at the end of or in between Neural Networks. They help to decide if the neuron would fire or not. For this network Rectified Linear Unit (ReLU). ReLU function is the most widely used activation function in neural networks today.
LeakyReLU and Softmax Activation Functions - Increses the accuracy of the model
Tanh and Sigmoid - Decreases the accurcay of the model

#### Part C - Cost function

A cost function is a measure of "how good" a neural network did with respect to it's given training sample and the expected output. It also may depend on variables such as weights and biases. A cost function is a single value, not a vector, because it rates how good the neural network did as a whole.

categorical_crossentropy - Cross-entropy will calculate a score that summarizes the average difference between the actual and predicted probability distributions for all classes in the problem. The score is minimized and a perfect cross-entropy value is 0.

Kullback Leibler Divergence, or KL Divergence for short, is a measure of how one probability distribution differs from a baseline distribution.


#### Part D - Epochs

The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset. An epoch is comprised of one or more batches
Decreasing the number of epochs has decreased the accuracy of the model and Increaseing the number of epoch will increase the accuracy of the model. 

#### Part E - Gradient estimation

Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.

Adagrad - Adaptive gradient is a parametric approcah which is related to how frequently parameters gets updated during training so less learning rate more update in parameter. This has increased the accuracy of the model.

sgd - Stochastic Gradient Descent, involves selecting random variable for x to find the minimum value of y. This gives an accuracy of 73%

#### Part F - Network Architecture

Increasing the number of layers will have a major impact of the prediction of the network. Increasing the number of layers has increased the prediction efficiency of the model

#### Part G - Network initialization

To build a machine learning algorithm, usually you’d define an architecture (e.g. Logistic regression, Support Vector Machine, Neural Network) and train it to learn parameters. Here is a common training process for neural networks: The initialization step can be critical to the model’s ultimate performance, and it requires the right method. I have used glorot_uniform and he_uniform for weight intialization

# References

1. https://www.investopedia.com/terms/d/deep-learning.asp
2. https://en.wikipedia.org/wiki/Deep_learning
3. https://machinelearningmastery.com/what-is-deep-learning/
4. https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
5. https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca
6. https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/
7. https://medium.com/@purnasaigudikandula/a-beginner-intro-to-convolutional-neural-networks-684c5620c2ce
8. https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
9. https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
10. https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7


```python

```
