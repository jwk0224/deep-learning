# 1. Convolutional Neural Networks

## Computer Vision Problems

Image Classification  
Object Detection  
Neural Style Transfer

## Convolution Operation

Dimension of image data is so big  
Convolution operation overcomes overfitting and memory overload

Libraries for convolution operation
- Python : conv-forward
- Tensorflow : tf.nn.conv2d
- Keras : Conv2D

Filter
- a matrix with values that can detect particular features of an image
- outputs a new image out of an original image
- values can be learned by training
- number of channels of filter must be same as that of input layer
- number of filters is the number of channels of output layer

Padding
- uses information of images in the edges
- keeps a size of image
- special type of paddings
	- Valid padding : no padding
	- Same padding : output size is same as input size

Strides
- step size of filter when conv forward

Output Size = ( (n + 2p - f) / s ) + 1

## Layers in Convolutional Network

1) Convolution (CONV) layer
   - n_h, n_w decrease as network go deeper
   - n_c increases as network go deeper


2) Pooling (POOL) layer
   - max or average pooling (commonly max pooling is used)
   - no parameters to learn
   - number of channels of output layer is same as input layer


3) Fully connected (FC) layer
   - same as general neural network

Common pattern
- CONV-POOL-CONV-POOL-FC-FC-FC-SOFTMAX

## Why Convolutions

1) Parameter sharing
   - One feature detector is used in all parts of image


2) Sparsity of connections
   - Each output value depends on small number of inputs

## Tensorflow Implementation

1) Sequential API
   - builds simple models with layer operations that proceed in a sequential order.


2) Functional API
   - builds models with non-linear topology, shared layers, layers with multiple inputs or outputs

# 2. Deep Convolutional Models: Case Studies

## CNN Case Studies

Classic networks

1) LeNet-5
   - Introduced one CNN architecture that works well


2) AlexNet
   - Similar to LeNet-5 but much bigger parameters
   - ReLU
   - Multiple GPUs
   - Local response normalization
   - Convinced community that DL works in computer vision


3) VGG-16 Net
   - Consistent CONV, MAX-POOL layer size
   - Relative uniformity of architecture was attractive to researchers
   - Large network with 16 layers that has weights (138M parameter)

Other networks

4) ResNet
   - Very deep plain networks don't work in practice because vanishing gradients make them hard to train
   - Skip connections help address the Vanishing Gradient problem
   - They also make it easy for a ResNet block to learn an identity function
   - There are two main types of blocks: The identity block and the convolutional block
   - Very deep Residual Networks are built by stacking these blocks together
   - Adding residual block doesn't hurt performance but only improves it if lucky
   - Residual block usually uses same convolution to preserve dimension


5) InceptionNet
   - 1x1 convolutions (= networks in networks) can shrinks the number of channels
   - Instead of choosing specific size of convolutions, train by trying multiple convolution operations and concatenate them
   - 1x1 convolution can reduce computational cost by factor of 10 without hurting performance
   - Inception network consists of a lot of inception blocks connected


6) MobileNet
   - Low computational cost at deployment
   - Useful for mobile and embedded vision applications
   - Depthwise separable convolution reduces computational cost
       1) Depthwise convolution : f x f x n_c filter, each filter for each single input layer
       2) Pointwise convolution : 1 x 1 x n_c filter
   - MobileNet v1 : 13 bottleneck blocks
   - MobileNet v2 : 17 bottleneck blocks
       - uses residual connections
       - uses expansion + depthwise + projection(=pointwise) convolutions


7) EfficientNet
   - automatically scales up or down neural networks for a particular device
   - look at one of the open source implementations of EfficientNet for r, d, w trade-off
       - r : input image resolution
       - d : depth of neural network
       - w : width of neural network

## Transfer Learning

Use pretrained weights and transfer that to a new task,  
instead of training the weights from scratch with random initialization

To adapt the classifier to new data
- Delete the top layer, add a new classification layer, and train only on that layer
- When freezing layers, avoid keeping track of statistics (like in the batch normalization layer)
- Fine-tune the final layers of your model to capture high-level details near the end of the network and potentially improve accuracy

Transfer learning is almost always used in computer vision,  
unless exceptionally large data set is secured

## Data Augmentation

Use data augmentation methods to increase data for training

1) Common augmentation method
   - Mirroring
   - Random cropping
   - Rotation
   - Shearing
   - Local warping


2) Color shifting method (change RGB value)

## State of Computer Vision

Little data <- Object detection - Image recognition - Speech recognition -> Lots of data

Two sources of knowledge
- Labeled data
- Hand engineered features/network architecture/other components

For little data : more hand-engineering, transfer learning  
For lots of data : less hand-engineering, simpler algorithms

Tips for doing well on benchmarks/winning competitions (but barely used for production system)
- Ensembling : train several networks independently and average their outputs
- Multi-crop at test time : run classifier on multiple versions of test images and average results

Use open source code
- Use architectures of networks published in the literature
- Use open source implementations if possible
- Use pretrained models and fine-tune on your dataset

# 3. Object Detection

## Detection Algorithms

Object recognition : input a picture and figure out what is in the picture

Object detection : put a bounding box around the object is found

Semantic segmentation : draw an outline around the object that is detected

## Object Localization

Image classification
- label the image
- target label y
	- c_1 : 0~1 (class 1 or not)
	- c_2 : 0~1 (class 2 or not)
	- c_3 : 0~1 (class 3 or not)

Classification with localization
- label the image and putting bounding box on the object
- target label y
	- p_c : 0~1 (there is an object or not)
	- b_x : center x position of an object
	- b_y : center y position of an object
	- b_h : height of an object
	- b_w : width of an object
	- c_1 : 0~1 (class 1 or not)
	- c_2 : 0~1 (class 2 or not)
	- c_3 : 0~1 (class 3 or not)

Detection
- label the image and putting bounding box on multiple objects
- target label y for each grid
	- p_c_1 : 0~1 (there is an object or not)
	- b_x_1 : center x position of an object
	- b_y_1 : center y position of an object
	- b_h_1 : height of an object
	- b_w_1 : width of an object
	- c_1 : 0~1 (class 1 or not)
	- c_2 : 0~1 (class 2 or not)
	- c_3 : 0~1 (class 3 or not)
	- p_c_2 : 0~1 (there is an object or not)
	- b_x_2 : center x position of an object
	- b_y_2 : center y position of an object
	- b_h_2 : height of an object
	- b_w_2 : width of an object
	- c_1 : 0~1 (class 1 or not)
	- c_2 : 0~1 (class 2 or not)
	- c_3 : 0~1 (class 3 or not)

## Landmark Detection

Landmark detection is one application of the object localization
- AR filters
- CG effects
- Pose detection

Neural network outputs X and Y coordinates of important points in image
- Labeled image set is required for training

## Object Detection

Uses sliding windows detection algorithm
1) divide an image with window (square box)
2) move over entire image with certain stride
3) See whether window contains an object or not
4) Iterate 1~3) with different window size

Computational cost can be reduced by convolutional implementation  
(Skip duplicated computations over mulitple windows)

## YOLO(You Only Look Once) Algorithm

One of the most effective object detection algorithms  
It runs very fast thanks to CNN and works for real time object detection

1) For each grid cell, get predicted bounding boxes (= ## of anchor boxes)
2) Get rid of low probability predictions
3) For each class, use non-max suppression to generate final predictions

Key ideas used are,

1. Bounding box predictions
   - Object is assigned to one cell that contains the mid point of object
   - Height and width can be greater than a grid size
   1) Divide an image with grid
   2) For each grid cell, run image classification and localization


2. Intersection over union
   - A function measuring the overlap between two bounding boxes
   - Used for evaluating object localization algorithm (ex. IoU > 0.5) and for Non-max suppression


3. Non-max suppression
   - A way to make sure that algorithm detects each object only once
   1) Discard all boxes with p_c <= 0.6
   2) While there are any remaining boxes :
      1) Pick the box with the largest p_c as a prediction
      2) Discard any remaining box with IoU >= 0.5 with the box in the previous step
   3) Iterate over each object class independently


4. Anchor boxes
   - A way to detect multiple objects in one grid cell
   - Anchor box : pre-defined shape that seems to cover the types of objects to detect
   - Object is assigned to one cell that contains the mid point of object and anchor box for the grid cell with highest IoU
   - How to choose anchor boxes
     - choose by hand
     - automatically choose by K-means algorithm
         - group together types of objects shapes
         - select anchor boxes that are most stereotypically representative of object

## Region Proposals

Picks just a few regions(windows) that makes sense to run conv-net classifier
- runs segmentation algorithm for region proposal

R-CNN (Regions with CNN)
- proposes regions
- classifies proposed regions one at a time
- outputs label and bounding box

Fast R-CNN
- proposes regions
- uses convolution implementation of sliding windows
   to classify all the proposed regions

Faster R-CNN
- uses convolutional network to propose regions
- YOLO is faster and more promising

## Semantic Segmentation with U-Net

Algorithm for many computer vision applications
- segmentation map for self-driving car
- medical imaging for diagnosis

It labels every single pixel individually with the appropriate class label

Transpose convolution : takes a small input and blow it up into larger output
1) Put filter on the output
2) Multiply filter values with input value
3) Sum output values that overlap

U-Net architecture
1) Normal convolutions compress the image
   - high level contextual information
2) Transpose convolutions blow up image size to original input size
3) Skip connections from earlier layers to matched later layers
   - low level detailed spatial information

# 4. Special Applications: Face recognition & Neural Style Transfer

## Face Recognition

Face verification : Is this the claimed person?
- Input image, name/ID
- Output whether the input image is that of the claimed person

Face recognition : Who is this person?
- Has a database of K persons
- Get an input image
- Output ID if the image is any of the K persons (or not recognized)

## One Shot Learning

Learning from one example to recognize the person again
- small training set is not enough to train neural network
- number of image data to recognize can be changed

Solution is to learn a similarity function

d(img1, img2) = degree of difference between images
- If d(img1, img2) ≤ τ -> same image
- If d(img1, img2) > τ -> different image

## Siamese Network

Good way to implement d(img1, img2)

1) Feed images into same neural network with same parameters
2) Forward propagate images and get an encoding vector
   - Pre-computing encodings can save a significant computation
3) Compare encodings to tell the difference

Paremeters of NN defines an encoding f(x(i))  
Learn parameters so that:  
- If x(i), x(j) are the same person, ||f(x(i)) - f(x(j))||^2 is small
- If x(i), x(j) are different person, ||f(x(i)) - f(x(j))||^2 is large

## Triplet Loss

One way to learn parameters of convnets for face recognition
- apply gradient descent on triplet loss fucntion

Triplets of images : Anchor(A), Positive(P), Negative(N)

L(A, P, N) = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + a, 0)
1) compute difference between two encodings
2) d(A, P) + a should be less than or equal to d(A, N)
   - a is a margin parameter

J = Σ_i L(A(i), P(i), N(i))

Multiple images of same person are required to train (A, P) pairs  
Triplets that's hard to train on should be chosen (= d(A, P) close to d(A, N))
- if randomly chosen, learning objective is too easily satisfied

## Binary Classification

Another way to learn parameters of convnets for face recognition
- treat face recognition just as a binary classification problem

Pairs of images : Same(1), Different(0)

y_hat = sigmoid(Σ_k w_i * |f(x(i))_k - f(x(j))_k| + b)
1) compute element-wise difference between two encodings
2) consider difference vector as feature x
3) train logistic regression parameters w_i, b

## Neural Style Transfer

Generates new image from content image and style image
- trains on the pixels of an image to make it look artistic
- it is not learning any parameters (no supervised learning)

G : Generated image = C : Content image + S : Style image

Make generated image G match the content of image C and the style of image S

J(G) = α*J_content(C, G) + β*J_style(S, G)
1) initialize G randomly
2) use gradient descent to minimize cost function J(G)

## Content Cost Function

Make generated image G match the content of image C

J_content(C, G) = 1/2 * || a[l] (C) - a[l] (G) ||^2
- difference of activation value between C and G in one chosen layer

The shallower layers
- tend to detect lower-level features such as edges and simple textures

The deeper layers
- tend to detect higher-level features such as more complex textures and object classes

Choose a layer to represent the content of generated image
- choose a layer in the middle of the network, neither too shallow nor too deep
- this makes network detect both higher-level and lower-level features

## Style Cost Function

Make generated image G match the style of image S
Define style as correlation between activations across channels in layer

J_style(S, G) = Σ_l λ[l] * J_style[l](S, G)
- sum of difference of correlation(=style) between S and G over chosen layers
- λ is a hyper parameter for each layer's weighting
	- if λ is large for deeper layer, generated image softly follows style image
	- if λ is large for shallower layer, generated image strongly follows style image

J_style[l](S, G) = 1/(2*n_h[l]*n_w[l]*n_c[l])^2 * Σ_kΣ_k' (G_kk'[l] (S) - G_kk'[l] (G))^2
- difference of correlation(=style) between S and G in layer l

G_kk'[l] (S) = Σ_iΣ_j a_ijk[l] (S)*a_ijk'[l] (S)
- correlation(=style) value of S in layer l
- 
G_kk'[l] (G) = Σ_iΣ_j a_ijk[l] (G)*a_ijk'[l] (G)
- correlation(=style) value of G in layer l

Style matrix = Gram matrix = G
- compares how similar two vectors
- computed by dot products -> np.dot(v_i, v_j)
- more similar two vectors -> larger the dot product

## 1D and 3D Generalizations

ConvNets ideas can be applied to not only 2D data but also 1D and 3D data
- 1D : medical signal
- 2D : images
- 3D : CT scan, movie