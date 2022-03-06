# Machine Learning

## Summary

Supervised learning
- Linear regression
- Logistic regression
- Neural networks
- SVMs

Unsupervised learning
- K-means clustering
- PCA
- Anomaly detection

Special applications/special topics
- Recommender systems
- Large scale machine learning

Advice on building a machine learning system
- Bias/variance
- Regularization
- Deciding what to work on next
- Evaluation of learning algorithms (precision, recall, F-score, training/cv/test)
- Learning curves (debugging)
- Error analysis
- Ceiling analysis

## Machine Learning

Definition:  
Field of study that gives computers the ability to learn without being explicitly programmed

Example:
- Database mining
- Applications can't program by hand
  - ex. Autonomous helicopter, handwriting recognition, NLP, computer vision
- Self-customizing programs
  - ex. Amazon, Netflix product recommendations
- Understanding human learning
  - ex. brian, real AI

Machine learning algorithms:
- Supervised Learning
- Unsupervised Learning
- Others: Reinforcement learning, recommender systems

## Supervised Learning

Right answers given for each example in the data
- Regression: Predict continuous valued output
- Classification: Predict discrete valued output

## Unsupervised Learning

Derive structure from data with little or no idea what results should look like
- Clustering: automatically group data
- Non-clustering: Cocktail Party Algorithm finds the structure in a chaotic environment

# Linear Regression

## Cost Function

= Squared Error Function = Mean squared error

Measuring the accuracy of the hypothesis function by using a cost function

## Gradient Descent Algorithm

The machine learning algorithm (vs normal equation)  
The way of estimating parameters in the cost function

Scales better to larger data sets than the normal equation

The Gradient Descent algorithm repeats until convergence
- If α is too small, gradient descent can be slow
- If α is too large, gradient descent can overshoot the minimum. It may fail to converge or even diverge

With fixed α gradient descent automatically takes smaller steps as the slope decrease

Linear Regression's cost function is convex function and only have a global optimum  
Linear Regression's gradient descent is called batch gradient descent because it uses every training set every time

## Vector

Vector is n x 1 matrix

## Feature Scaling

Make sure features are on a similar scale to make gradient descent faster

Get every feature into approximately a -1 <= x(i) <= 1 range  
= feature value / feature range (max - min)

## Mean Normalization

Replace x(i) to make features approximately zero mean  
= (x(I) - avg) / std or range (max - min)

## Polynomial Regression

Hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can change the behavior or curve of hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For polynomial regression, feature scaling is important.

## Gradient Descent vs Normal Equation Algorithm

When n exceeds 10,000, consider Gradient Descent

Gradient Descent
- Need to choose alpha
- Needs many iterations
- O(kn^2)
- Works well when n is large

Normal Equation
- No need to choose alpha
- No need to iterate
- O(n^3), need to calculate inverse of X(T)X
- Slow if n is very large

## Normal Equation Noninvertibility

Common cause for noninvertibility
- Redundant features, where two features are very closely related. Delete linearly dependent features
- Too many features (ex. M <= n). Delete some features or use regularization

# Logistic Regression

Logistic Regression is a classification algorithm

Linear Regression doesn't work well for classification because classification is not actually a linear function

Sigmoid function = Logistic function = g(x)

hθ(x) = g(θTx)  
y is the probability that output is 1 with given x

## Decision Boundary

The line that separates the area where y = 0 and where y = 1.

It is created by hypothesis function regardless of data set.

## Advanced Optimization

More sophisticated and faster ways to optimize θ than gradient descent
- Conjugate gradient
- BFGS
- L-BFGS

Need to provide two functions
- Cost function: Code to compute J(θ)
- Gradient: Code to compute derivative of J(θ)

## Multi-class Classification

One-vs-all

Train a logistic regression classifier hθ(x) for each class to predict the probability that y = i  
To make a prediction on a new x, pick the class that maximizes hθ(x)

## Overfitting

Hypothesis function fits the available data but does not generalize well to predict new data

Underfitting = high bias  
Overfitting = high variance

To address overfitting problem
- Reduce the number of features
- Manually select which features to keep
- Use a model selection algorithm
- Regularization
- Keep all the features, but reduce the magnitude of parameters
  - (Works well when we have a lot of slightly useful features)

The λ, or lambda, is the regularization parameter for extra summation to hypothesis function.

# Regularization

Penalize all parameters to become somewhat reduced values

Linear Regression
- Gradient Descent: adding λ⋅θ^2 term at the end of cost function when θ(1)~θ(n)
- Normal Equation: adding λ⋅L term inside inverting matrix (solving invertibility problem)

Logistic Regression
- Gradient Descent: adding λ⋅θ^2 term at the end of cost function when θ(1)~θ(n)
- Advanced Optimization: same as above

# Neural Networks

Complex nonlinear hypothesis with large number of features
- overfitting problem
- computationally expensive

Neural Networks (vs Linear Regression, Logistic Regression) is needed

Neurons are basically computational units that take inputs (dendrites)  
as electrical inputs (called "spikes") that are channeled to outputs (axons)

Forward propagation
- input layer(data) - hidden layer(activation units) - output layer(hypothesis function)
- sigmoid (logistic) activation function
- parameters = weights
- each layer gets its own matrix of weights, Θ(j)

## Cost function of Neural Networks

Sum of output units of output layer (each unit has same cost function as logistic regression)  
plus Regularization term (sum of entire parameters in the network except bias unit)

## Back Propagation Algorithm

Backpropagation is neural-network terminology for minimizing cost function,  
just like gradient descent in logistic and linear regression.

## Unrolling Parameters

thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]

Theta1 = reshape(thetaVector(1:110),10,11)  
Theta2 = reshape(thetaVector(111:220),10,11)  
Theta3 = reshape(thetaVector(221:231),1,11)

## Gradient Checking

Numerically compute approximate gradient and see if it is same or similar to actual gradient  
to assure that back propagation works as intended

Turn off gradient checking after confirmation because it is computationally very expensive

## Random Initialization

If we initialize all theta weights to same value (like 0) with neural networks,  
when we backpropagate, all nodes will update to the same value repeatedly

Instead we can randomly initialize our weights - Symmetry Breaking

## Neural Network Architecture

Number of input units = dimension of features x(i)  
Number of output units = number of classes  
Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)

Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

## Training a Neural Network

1) Randomly initialize the weights
2) Implement forward propagation to get hΘ(x(i)) for any x(i)
3) Implement the cost function
4) Implement backpropagation to compute partial derivatives
5) Use gradient checking to confirm that your backpropagation works. Then disable gradient checking
6) Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta

# Evaluating the Hypothesis

1) Divide data into training(60%)/cross validation(20%)/test Set(20%)
2) Create list of λ
3) Create a set of models with different degrees or any other variants
4) Iterate through the λs and for each λ go through all the models to learn Θ with training set
5) Compute the cross validation error(without λ) using the learned Θ
6) Select the best combo that produces the lowest cross validation error
7) Using the best combo Θ and λ, estimate generalization error with test set

## Diagnosing Bias vs Variance

High Bias means Underfitting  
High Variance means Overfitting  
High training set error & High test set error means High Bias  
Low training set error & High test set error means High Variance

We can use Learning Curves to diagnose Bias/Variance
- Y-axis: Train error & Test error
- X-axis: Training set size

In high bias, getting more training data will not (by itself) help much  
In high variance, getting more training data is likely to help

## What to Do to Improve Model

Getting more training examples: Fixes high variance  
Trying smaller sets of features: Fixes high variance  
Adding features: Fixes high bias  
Adding polynomial features: Fixes high bias  
Decreasing λ: Fixes high bias  
Increasing λ: Fixes high variance

## Diagnosing Neural Networks

No. of Neural Networks parameters: No. of layers and No. of units in each layer

Fewer parameters is prone to underfitting and computationally cheaper  
More parameters is prone to overfitting and computationally expensive
- Use regularization (increase λ) to address the overfitting

More parameters with regularization is better than fewer parameters

## Error Analysis

The recommended approach to solving machine learning problems is to:

1) Start with a simple algorithm, implement it quickly, and test it early on cross validation data
2) Plot learning curves to decide if more data, more features, etc. are likely to help
3) Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made
4) Try new feature and get a numerical value for the error rate
5) Decide whether to keep the new feature or not

## Precision & Recall

Accuracy = (true positives + true negatives) / (total examples)

For skewed classes, Precision & Recall is a better way to evaluate learning algorithm than classification error or classification accuracy  
The algorithm cannot cheat on both metrices

Precision = (true positives) / (true positives + false positives)  
Recall = (true positives) / (true positives + false negatives)

## F Score

Precision and Recall have a trade-off relationship depending on the threshold  
One way to decide an optimal threshold is to use F score

F score = (2 * precision * recall) / (precision + recall)

1) Try a range of different values of thresholds for cross validation set
2) Choose threshold that produces highest F score (both P&R should be high for this)

## Large Data Rationale

Collect more data can be useful when:

Using feature x that has sufficient information to predict y accurately
- (Can a human expert confidently predict y with x?)

Using a low bias algorithm with many parameters  
Using a very large training set resulting in low variance

# Support Vector Machine

Powerful classification algorithm for data with small features and intermediate data size

SVM uses alternative cost function from Logistic Regression  
SVM is called Large Margin Classifier because of the decision boundary

If y = 1, we want θ_Tx >= 1 (not just >= 0)  
If y = 0, we want θ_Tx <= -1 (not just <= 0)

## SVM Cost Function

= min C * sum[ y_i * cost_1(θ_T * f_i) + (1 - y_i) * cost_0(θ_T * f_i) ] + 1/2 * sum(θ_j^2)  
- C : regularization term(= 1/λ) is moved to the front part of function instead of the rear
- cost_1 : (when y = 1) the cost becomes 0 when θ_T * x_i >= 1
- cost_0 : (when y = 0) the cost becomes 0 when θ_T * x_i <= -1
- f_i : new feature computed by kernel

To use SVM, need to choose parameter C and Kernel

## SVM Kernel

SVM uses kernel to create complex non-linear boundary (better than polynomial features)

Kernel is similarity function that computes new feature(f) that is the distance between a training example(x) and a landmark(l)

Training examples are used for landmarks
- If x is close to l, f becomes closer to 1
- If x is far from l, f becomes closer to 0

1) No(=Linear) kernal :  
   - Simply using x_i instead of f_i


2) Gaussian kernel :
   - f_i = exp(- ||x - l_i||^2/2σ^2)  
   - Need to choose σ^2  
   - Need to perform feature scaling


3) Polynomial kernel, String kernel, chi-square kernel, histogram intersection kernel …
   - SVM predicts
   - 1 when θ0 + θ1f1 + θ2f2 + θ3*f3 + … >= 0  
   - 0 otherwise

## SVM Parameters

Large C : Low bias, High variance (Try to classify every single example correctly)  
Small C : High bias, Low variance (Ignore outliers)

Large σ^2 : High bias, Low variance  
Small σ^2 : Low bias, High variance

## Logistic Regression vs SVM

n = number of features  
m = number of training examples

If n is large (relative to m): Use logistic regression, or SVM without a kernel  
If n is small, m is intermediate: Use SVM with Gaussian kernel  
If n is small, m is large: Create/add more features, then use logistic regression or SVM without a kernel

Neural network likely to work well for most of these settings, but may be slower to train.

# Clustering

Applications of clustering
- Market segmentation
- Social network analysis
- Organize computing clusters
- Astronomical data analysis

## K-means algorithm

K-means is a clustering algorithm

1) Randomly initialize K cluster centroids


2) Repeat
   - Cluster assignment step :
     - for i = 1 to m
     - c_i = index of cluster centroid closest to x_i (from 1 to K)
   - Move centroid step :
     - for k = i to K
     - μ_k = average(mean) of points assigned to cluster k

Cost function(distortion) = min 1/m * sum(||x_i - μc_i||^2)  
Sum of distance between every example and centroid that example is assigned to  
Cost must decrease every iteration

## Random Initialization

Randomly pick K training examples and use them as centroids (K < m)

Randomly initialized clustering may end up with local optima  
Get clusters with random initialization multiple times and pick a cluster with lowest cost

## Choosing the Number of Clusters

Normally, as the number of clusters increases, the cost of cluster decreases

Elbow method checks the cost of cluster as the number of clusters increases  
Finds K where the cost doesn't meaningfully decrease anymore

Ultimately, choose K that performs best for the purpose

# Dimensionality Reduction

Applications of dimensionality reduction

1) Compression (choose k by % of variance retained)
   - Reduce memory/disk needed to store data
   - Speed up learning algorithm


2) Visualization (k = 2 or 3)
   - Understand data better by plotting on 2D or 3D dimensional space

## Principal Component Analysis

PCA is a dimensionality reduction algorithm
PCA finds lower dimensional surface by minimizing average squared projection error

1) Perform mean normalization and (optional) feature scaling


2) Compute covariance matrix
   - Sigma = 1/m * sum(x_i * x_i')


3) Compute eigenvectors
   - [U, S, V] = svd(Sigma) -> Singular Value Decomposition


4) Take first k columns of vector U
   - U_reduce = U(:, 1:k)


5) Compute reduced feature z
   - z = U_reduce' * x

## Reconstruction from Compressed Representation of PCA

z(i) = U_reduce' * x(i)  
X_approx(i) = U_reduce * z(i)

## Choosing No. of Principal Components k

Choose k to be smallest value so that :  
- average squared projection error / total variation in the data <= 0.01 or 0.05
- It is said that 99% or 95% of variance is retained

1/m * sum(||x_i - x_i_approx||^2) / 1/m * sum(||x_i||^2) <= 0.01 or 0.05  
or  
[U, S, V] = svd(Sigma)

Choose smallest value of k for which  
sum~k(S_ii)/sum~m(S_ii) >= 0.99 or 0.95

# Anomaly Detection

Unsupervised learning algorithm for detecting anomaly

1) Choose features that might be indicative of anomalous example


2) Compute parameters μ, σ for each feature with examples x_i
   - Assume that each feature is Gaussian distribution


3) Compute p(x) with the formula p for new example x
   - p(x) = p(x_1)p(x_2)…p(x_n)


4) Select threshold ε with the highest F score on a cross validation set


5) Anomaly if p(x) < ε

Using multivariate Gaussian distribution
- Model p(x) all at once instead of multiplying p(x_i)
- μ is a vector of mean of each feature x
- Σ is a covariance matrix of each feature x
- Automatically captures correlations between features but computationally more expensive
- m > n is required because Σ should be invertible 

Choosing what features to use
- Transform non-gaussian features into Gaussian using log, exponential, squared root
  - (Highly recommended but it's okay if not)
- Extract meaningful features from anomalous examples by doing an error analysis
- Create features that can have unusually large or small values by combining existing features

## Evaluation of Anomaly Detection Algorithm

Train with normal examples only  
CV, Test with anomalous examples

For 10,000 normal example, 20 anomalous example
- Training set : 6,000 (normal)
- Cross Validation set : 2,000 (normal), 10 (anomalous)
- Test set : 2,000 (normal), 10 (anomalous)

Precision/Recall, F-score can be good evaluation metrics  
Do not use classification accuracy because data is skewed

## Anomaly Detection vs Supervised Learning

Anomaly Detection
- Very small number of positive examples (0-20 is common)
- Type of anomalies are various and future anomalies may also be a new type

Supervised Learning
- Large number of positive examples
- Type of positive examples can be generalized and the future positive examples is likely to be a similar type

# Recommender Systems

A subclass of information filtering system  
that seeks to predict the "rating" or "preference" a user would give to an item (special application of machine learning)

## Collaborative Filtering

An algorithm that learns for itself what features to use (feature learning)

Given a dataset that consists of a set of ratings produced by some users on some contents,  
the objective is to learn the parameters(x, θ) that produce the best guess on missing rating

Both x (content feature) and θ (user preference on feature) parameters are simultaneously updated to minimize cost function

1) Initialize x_1~x_nm, θ_1~θ_nu to small random values
2) Minimize J(x_1~x_nm, θ_1~θ_nu) using gradient descent (or advanced optimization algorithm)
3) Predict user's rating on contents : θ'x

Find most related contents : smallest ||x_i - x_j||

## Mean Normalization for Collaborative Filtering

If an user has not rated any contents, predicted ratings y for the user is all 0

With mean normalization, y becomes mean of existing ratings from all users (better than 0)

Use Y - μ instend of Y for all existing ratings, predict user's rating on contents with θ'x + μ
- μ = mean of existing ratings from all users
- Y = ratings

# Large Scale Machine Learning

High bias problem : add more features and etc  
High variance problem : get more data and etc

Two ways of learning with large datasets
1) Stochastic gradient descent
2) Map-reduce

## Stochastic Gradient Descent

Randomly shuffle dataset and compute cost per every training example

With smaller steps, getting sense early of how learning goes is possible
- (Batch) gradient descent: Use all m examples in each iteration
- Stochastic gradient descent: Use 1 example in each iteration
- Mini-batch gradient descent: Use b examples in each iteration

To check convergence, plot average of cost per every n iterations (ex. n = 1,000)

Not standard but slowly decreasing α over time can help θ converge
- ex. α = const1 / (iterationNumber + const2)

## Online Learning

Model learns every time new data is generated by an user (and discards data)

Large companies running website use online learning regarding click data and etc  
A Model can automatically adapt to changing user behavior or trend

## Map-reduce Data Parallelism

1) Express learning algorithm as computing sums of functions over the training set
2) Split data and compute each part on multiple computers or cores parallelly
3) Combine computed results on a central server

Software library may support using map-reduce as if we're dealing with one data set  
Network speed between machines may slow down parallel computation speed a little

# End-to-end Application Example of Machine Learning

## Machine learning pipeline

A system with many stages/components, several of which may use machine learning

Photo OCR pipeline
1) Text detection (sliding window - binary classification)
2) Character segmentation (1D sliding window - binary classification)
3) Character classification
4) Spelling correction

## Getting Lots of Data

1) Make sure to have a low bias classifier before expanding efforts
2) How much work would it be to get 10x as much data as we currently have?
   - Artificial data synthesis
   - Generating new data from scratch (ex. downloads fonts…)
   - Amplifying existing data by distortion (ex. noise on text, image, audio…)
   - Collect/label by myself
   - Crowd source (ex. Amazon Mechanical Turk)

Distortion should be representation of the type of noise/distortions in the test set  
Adding purely random/meaningless noise to data does not help

## Ceiling Analysis

The method for deciding what part of the pipeline should be the focus for improvement

Start from the fore part of the pipeline to the end  
Compute the final accuracy when each module produces the perfect result (manually)  
Focus on the module that has the most upside potential