# deep-learning

Exercise projects on basic deep learning algorithms from Deep Learning Specialization course by Andrew Ng on Coursera

## dl_ex1_Python Basics with Numpy

- Be able to use iPython Notebooks
- Be able to use numpy functions and numpy matrix/vector operations
- Understand the concept of "broadcasting"
- Be able to vectorize code

## dl_ex2_Logistic Regression as a Neural Network

- Build the general architecture of a learning algorithm, including:
  - Initializing parameters
  - Calculating the cost function and its gradient
  - Using an optimization algorithm (gradient descent)
- Gather all three functions above into a main model function, in the right order

## dl_ex3_Planar Data Classification with One Hidden Layer

- Implement a 2-class classification neural network with a single hidden layer
- Use units with a non-linear activation function, such as tanh
- Compute the cross entropy loss
- Implement forward and backward propagation

## dl_ex4_Building Your Deep Neural Network: Step by Step

- Use non-linear units like ReLU to improve your model
- Build a deeper neural network (with more than 1 hidden layer)
- Implement an easy-to-use neural network class

## dl_ex5_Deep Neural Network for Image Classification: Application

- Build and apply a deep neural network to supervised learning

## dl_ex6_Initialization

- Speed up the convergence of gradient descent
- Increase the odds of gradient descent converging to a lower training (and generalization) error

## dl_ex7_Regularization

- Use regularization in your deep learning models

## dl_ex8_Gradient Checking

- Implement and use gradient checking

## dl_ex9_Optimization Methods

- Speed up learning and get a better final value for the cost function using more advanced optimization methods

## dl_ex10_TensorFlow Tutorial

- Initialize variables
- Start your own session
- Train algorithms
- Implement a Neural Network

## cnn_ex1_Convolutional Neural Networks: Step by Step

- Explain the convolution operation
- Apply two different types of pooling operation
- Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
- Build a convolutional neural network

## cnn_ex2_Convolutional Neural Networks: Application

- Build and train a ConvNet in TensorFlow for a binary classification problem
- Build and train a ConvNet in TensorFlow for a multiclass classification problem
- Explain different use cases for the Sequential and Functional APIs

## cnn_ex3_Residual Networks

- Implement the basic building blocks of ResNets in a deep neural network using Keras
- Put together these building blocks to implement and train a state-of-the-art neural network for image classification
- Implement a skip connection in your network

## cnn_ex4_Transfer Learning with MobileNetV2

- Create a dataset from a directory
- Preprocess and augment data using the Sequential API
- Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
- Fine-tune a classifier's final layers to improve accuracy

## cnn_ex5_Car Detection with YOLO

- Detect objects in a car detection dataset
- Implement non-max suppression to achieve better accuracy
- Implement intersection over union as a function of NMS
- Create usable bounding box tensors from the model's predictions

## cnn_ex6_Image Segmentation with U-Net

- Build your own U-Net
- Explain the difference between a regular CNN and a U-net
- Implement semantic image segmentation on the CARLA self-driving car dataset
- Apply sparse categorical crossentropy for pixelwise prediction

## cnn_ex7_Face Recognition

- Differentiate between face recognition and face verification
- Implement one-shot learning to solve a face recognition problem
- Apply the triplet loss function to learn a network's parameters in the context of face recognition
- Explain how to pose face recognition as a binary classification problem
- Map face images into 128-dimensional encodings using a pretrained model
- Perform face verification and face recognition with these encodings

## cnn_ex8_Art Generation with Neural Style Transfer

- Implement the neural style transfer algorithm
- Generate novel artistic images using your algorithm
- Define the style cost function for Neural Style Transfer
- Define the content cost function for Neural Style Transfer

## sqm_ex1_Building Your Recurrent Neural Network: Step by Step

- Define notation for building sequence models
- Describe the architecture of a basic RNN
- Identify the main components of an LSTM
- Implement backpropagation through time for a basic RNN and an LSTM
- Give examples of several types of RNN

## sqm_ex2_Character Level Language Model: Dinosaurus Island

- Store text data for processing using an RNN
- Build a character-level text generation model using an RNN
- Sample novel sequences in an RNN
- Explain the vanishing/exploding gradient problem in RNNs
- Apply gradient clipping as a solution for exploding gradients

## sqm_ex3_Improvise a Jazz Solo with an LSTM Network

- Apply an LSTM to a music generation task
- Generate your own jazz music with deep learning
- Use the flexible Functional API to create complex models

## sqm_ex4_Operations on Word Vectors

- Explain how word embeddings capture relationships between words
- Load pre-trained word vectors
- Measure similarity between word vectors using cosine similarity
- Use word embeddings to solve word analogy problems such as Man is to Woman as King is to __.