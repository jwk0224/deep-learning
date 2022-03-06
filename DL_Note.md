# Neural Networks and Deep Learning

## Deep Learning

DL's most application is for supervised learning  
DL works with both structured and unstructured data

Traditional algorithms' performance doesn't increase with more data at some point  
DL's performance keeps increasing with more data

Scale drives DL progress
- Data
- Computation
- Algorithms

Whenever possible, avoid exploit for-loops

## Activation Functions

Activation function of a node defines the output of that node given an input or set of inputs

Different activation functions can be used for layers(units) in one neural network  
Slope of activation functions is derivative. So it affects the speed of gradient descent learning  
When slope is close to 0, it slows down the learning

If linear function is used for activation function, it produces linear function  
No matter how deep is the network, there will be no hidden unit (just linear regression)

1) Sigmoid
   - Never use this except for the binary classification (output layer)
   - Natural to use when the output value should be between 0 and 1


2) Tanh
   - Always superior to Sigmoid
   - The mean of its output is closer to zero, and so it centers the data better for the next layer
   - Sigmoid & Tanh derivative can be small (so slow) when z value is very large or very small


3) ReLU(Rectified Linear Unit)
   - Default choice
   - When doesn't know what to use for hidden unit, use ReLU
   - Computationally faster than Sigmoid
   - Derivative is 1 when z value > 0, 0 when z value < 0


4) Leaky ReLU
   - Feel free to try
   - ReLU but derivative is slightly decreasing when z value < 0
   - Usually works better than ReLU but not used much in practice

## Random Initialization

Random initialization solves symmetry breaking problem (parameters are not learned)

Initializing parameters with all 0 leads to symmetry breaking problem  
Each unit in the same layer affects same to the output node (no point of many nodes)

So randomly initialize w (b is okay with 0)  
For Sigmoid or Tanh, multiply small number (ex. 0.01) to make sure the value is not too large

# Practical Aspects of Deep Learning

## Train / Dev / Test Set

Traditionally, 70/30, 60/20/20 ratio were standards  
Recently with big data, dev, test set can be much smaller

Make sure train, dev, test set come from same data distribution (ex. same image quality)

## Bias and Variance

Low train set error & High dev set error : high variance  
High train set error & High dev set error : high bias  
High train set error & Even higher dev set error : high bias & high variance  
Low train set error & Low dev set error : low bias & low variance

‘High' depends on comparison to the optimal error value (ex. how well human can do)

## Basic Recipe for Machine Learning

1) High Bias? (training set performance)  
   - Bigger network -> Reduce bias, no trade off for variance (if regularization is appropriate)
   - Train longer
   - (Try other NN architecture)


2) High Variance? (dev set performance)
   - More data -> Reduce variance, no trade off for bias
   - Regularization
   - (Try other NN architecture)


3) Low Bias & Low Variance

## Regularization

Regularization reduces overfitting

Normally only w is regularized because b is just single number(insignificant)  
Smaller network (due to regularization effect) tends to be simpler and lower variant

1) L2 regularization (= weight decay)
   - Reducing the weight effect of the parameter
   - Most common regularization method
     - Add lamda/2m*sum(||w||^2) to cost function J  
     - ||w||^2 is called Frobenius norm of a matrix (F)


2) L1 regularization
   - Reducing the weight effect of the parameter
   - w gets more sparse (many zeroes)
     - Add lamda/2m*sum(||w||) to cost function J


3) Dropout regularization
   - Eliminating nodes randomly in a certain ratio
   - Inverted Dropout is most common method
   - It's not used at test time because we don't want the prediction results to be random
     - dl = np.random.randn(al.shape[0], al.shape[1]) < keep-prob
     - al = np.multiply(al, dl)
     - al /= keep-prob -> inverted dropout (keep amount of total value for next layer same)
     
Can't rely on any one feature because it can go away randomly  
So it spread out weights generating simliar effect to L2 regularization
- keep-prob may be set by layer
- keep-prob 1.0 means dropout is not applied
- keep-prob is usually 1.0 for input layer


4) Data augmentation  
   - Increasing data by flipping, rotating, cropping image


5) Early stopping
   - Inspect training and dev set error(cost) over no. of iteration  
   - Stop at some point before overfitting happens
     - Good that we don't have to try many times like L2 regularization
     - Bad that cost optimization and overfitting prevention tasks are mixed

## Normalization

Normalizing inputs can speed up the learning

If features have similar scale, gradient descent can go much faster to optimize  
Just do this as it never harms

1) Subtract mean : x = x - mu
2) Normalize variance : x = sigma^2 / x 

Normalize training set and test set with same mu and sigma^2

## Weight Initialization

Weight initialization can be a partial solution for vanishing & exploding gradient issue

When nn is deep, output value becomes exponentially big or small  
It's called vanishing & exploding gradient issue and it makes learning difficult

So for bigger n (= no. of weight parameters), make the variance of w[i] smaller

For Relu  
- w[l] = np.random.randn(shape)*np.sqrt(2/n^[l-1])
- So that Var(w[i]) = 2/n

For Tanh (= Xavier Initialization)  
- w[l] = np.random.randn(shape)*np.sqrt(1/n^[l-1])
- So that Var(w[i]) = 1/n

## Gradient Checking

Grad check can help checking whether back prop works right and whether there's bug

Difference = dTheta_approx - dTheta / dTheta_approx + dTheta
- Derivative approximation = f(theta + epsilon) - f(theta - epsilon) / 2*epsilon
- Approximation error = epsilon^2

If difference is less than 10^7, we can be quite confident that gradient computation is correct

Tips
- Use gradient check only for debugging (not training)
- If grad check failed, find certain component that shows different value difference
- Remember to include regularization terms if exit
- Turn off drop out when grad check
- If grad check false alarms due to random initialization, run some training and do it again

# Optimization Algorithms

## Mini-batch Gradient Descent

Starting gradient descent before finishing processing entire training set can get faster result

Every one in deep learning uses mini-batch gradient descent when training a large data set  
Shuffle and partition the entire batch into mini-batch

(1 epoch : pass through training set)

Mini-batch size = m (= batch gradient descent)
- Too long per iteration

Mini-batch size = 1 (= stochastic gradient descent)
- Lose speed up from vectorization

Mini-batch size = Not too big/small
- Fastest learning (vectorization, making progress without processing entire training set)

Choosing mini-batch size
- Small training set (m <= 2,000) : use batch gradient descent
- Large training set (m > 2,000) : typical mini-batch size is 64, 128 256, 512
- Make sure mini-batch fits in CPU/GPU memory size

## Gradient Descent with Momentum

Momentum almost always works faster than the standard gradient descent algorithm

It computes an exponentially weighted average of gradients and use them to update weights  
Gradients efficiently move toward minimum faster (average out oscillation, keep direction)

𝑣_𝑑𝑊 = 0, 𝑣_𝑑𝑏 = 0  
On iteration 𝑡:
- Compute 𝑑𝑊, 𝑑𝑏 on the current mini-batch (or just batch)
- 𝑣_𝑑𝑊 = 𝛽*𝑣_𝑑𝑊 + (1 − 𝛽)*𝑑𝑊
- 𝑣_𝑑𝑏 = 𝛽*𝑣_𝑑𝑏 + (1 − 𝛽)*𝑑𝑏
- 𝑊 = 𝑊 − 𝛼𝑣_𝑑𝑊, 𝑏 = 𝑏 − 𝛼𝑣_𝑑𝑏

Hyperparameters
- learning rate 𝛼
- exponentially weighted average 𝛽 : 0.9 → common choice (average over last 10 iteration's gradient)

Exponentially weighted average  
- Average of accumulating values with exponentially large emphasis on recent n values
- Advantages
  - Takes very little memory (keep only one real number and overwrite it)
  - Computationally efficient
  - Requires one line of code
- Bias correction makes it more accurate during the initial phase of estimate
  - v_dW = v_dW/(1 - 𝛽^t)
  - Put more emphasis on initial values
  - But as it goes toward end values, emphasis becomes close to zero
  - Mostly not required because just waiting for initial section to pass may be enough

## RMSprop

Root Mean Square prop can speed up gradient descent

s_𝑑𝑊 = 0, s_𝑑𝑏 = 0  
On iteration 𝑡:  
Compute 𝑑𝑊, 𝑑𝑏 on the current mini-batch (or just batch)
- s_𝑑𝑊 = 𝛽*s_𝑑𝑊 + (1 − 𝛽)*𝑑𝑊^2
- s_𝑑𝑏 = 𝛽*s_𝑑𝑏 + (1 − 𝛽)*𝑑𝑏^2
- 𝑊 = 𝑊 − 𝛼*(dW/sqrt(s_𝑑𝑊)), 𝑏 = 𝑏 − 𝛼*(db/sqrt(s_𝑑𝑏))

In dimensions where getting oscillations, derivative is divided by a much larger number  
Dumping out oscillations and allowing to use larger learning rate

## Adam Optimization Algorithm

Adaptive Moment Estimation combines the effect of Momentum together with RMSprop  
Commonly used learning algorithm that is proven to be very effective

s_𝑑𝑊 = 0, s_𝑑𝑏 = 0, s_𝑑𝑊 = 0, s_𝑑𝑏 = 0  
On iteration 𝑡:  
Compute 𝑑𝑊, 𝑑𝑏 on the current mini-batch (or just batch)  
- 𝑣_𝑑𝑊 = 𝛽_1*𝑣_𝑑𝑊 + (1 − 𝛽_1)*𝑑𝑊
- 𝑣_𝑑𝑏 = 𝛽_1*𝑣_𝑑𝑏 + (1 − 𝛽_1)*𝑑𝑏
- v_dW_corrected = v_dW/(1 - 𝛽_1^t) → Bias correction
- v_db_corrected = v_db/(1 - 𝛽_1^t)


- s_𝑑𝑊 = 𝛽_2*s_𝑑𝑊 + (1 − 𝛽_2)*𝑑𝑊^2
- s_𝑑𝑏 = 𝛽_2*s_𝑑𝑏 + (1 − 𝛽_2)*𝑑𝑏^2
- s_dW_corrected = s_dW/(1 - 𝛽_2^t)
- s_db_corrected = s_db/(1 - 𝛽_2^t)


- 𝑊 = 𝑊 − 𝛼*(v_dW_corrected/(sqrt(s_dW_corrected) + epsilon))
- 𝑏 = 𝑏 − 𝛼*(v_db_corrected/(sqrt(s_𝑑𝑏_corrected) + epsilon))

Hyperparameters : 𝛼 (learning rate) ,𝛽 (exponentially weighted average)
- 𝛼 → tune
- 𝛽_1 : 0.9 → use default choice
- 𝛽_2 : 0.999 → use default choice
- epsilon : 10^-8 → use default choice (prevents dividing by 0 error)

## Learning Rate Decay

Learning rate decay speeds up learning by slowly reducing learning rate over time

It takes smaller steps as learning approaches converges,

𝛼 = 1/(1 + decay_rate*epoch_num) * 𝛼_0
- 𝛼_0 : initial learning rate

Exponential decay, discrete staircase, manual decay are other methods  
Priority is lower than other hyperparameters as setting initial learning rate has a huge impact

## Local Optima in Deep Learning

In deep learning, it is unlikely to get stuck in bad local optima

In a function of high dimensional space, it's impossible for all directions to have a zero gradient  
Most points of zero gradient in a cost function are saddle points (not local optima)

Plateaus can make learning slower and algorithms such as Adam can help speed up

# Hyperparameter Tuning, Batch Normalization and Programming Frameworks

## Tuning Process

Priority
1) learning rate alpha
2) momentum beta
3) no. of hidden units
4) mini-batch size
5) no. of layer
6) learning rate decay
7) adam beta1, beta2, epsilon → use default

Try random values over combination of various parameters  
Don't use a grid for combination as important parameter can be wasted  
Coarse to find is frequently used to search certain areas more densely

Appropriate scaling for hyperparameters helps sampling 'uniformly' at random over the range  
ex) Learning rate can be log-scaled to uniformly test parameter value over the range

## Hyperparameter Tuning in Practice

Two hyperparameter tuning approachs

1) Baby sitting one model
   - A huge data set but not a lot of computational resources (CPUs and GPUs)
   - Gradually watch learning curve and try different parameters


2) Training many models in parallel
   - A huge data set with a lot of computational resources (CPUs and GPUs)
   - Try a lot of different hyperparameters and see what works

Depending on application, parallel approach may not be possible  
ex) Online advertising settings and computer vision applications have so much data

## Batch Normalization

Normalizing values in hidden layer helps learning in a NN (like normalizing input)

𝜇 = 1/𝑚 ∑𝑧^(𝑖)  
𝜎^2 = 1/𝑚 ∑(𝑧^(𝑖) − 𝜇)^2  
𝑧_norm^(𝑖) = (𝑧^(𝑖) − 𝜇)/√(𝜎^2 + 𝜀)  
𝑧 ̃ ^(𝑖) = 𝛾*𝑧_norm^(𝑖) + 𝛽

In batch normalization, hidden unit z^(i) have standardized mean and variance  
Fixed mean 0, variance 1 may not be optimal to take advantage of the nonlinearity  
So mean and variance are controlled by two explicit parameters gamma and beta

With batch normalization, earlier layers don't get to shift around as much  
Because they're constrained to have the same mean and variance  
So this makes the job of learning on the later layers easier

It has a slight regularization effect as a second effect

At test time, we might need to process a single example at a time (not a mini-batch)  
Exponentially weighted average are used to get a rough estimate of mu and sigma squared

## Softmax

Softmax is an activation function for multi-class classifier

a[l] = e^z[l]/sum(e^z[l])

It outputs the vector of the probability of each class is predicted (total 100%)

## Deep Learning Framework

- Caffe/Caffe2
- CNTK
- DL4J
- Keras
- Lasagne
- mxnet
- PaddlePaddle
- TensorFlow
- Theano
- Torch

Choosing deep learning frameworks
- Ease of programming (development and deployment)
- Running speed
- Truly open (open source with good governance)

As of 2019, Google launched TensorFlow 2  
TensorFlow 2 borrowed its syntax from Keras  
Keras is a high level library that can operate on top of TensorFlow 1 and other deep learning libraries

# Structuring Machine Learning Projects

## Orthogonalization

Orthogonalization is to verify the algorithms independently from one another  
4 assumptions that needs to be true and orthogonal for a supervised learning system design

1) Fit training set well in cost function
   - Use of a bigger neural network might help
   - Switching to a better optimization algorithm might help


2) Fit development set well on cost function
   - Regularization or using bigger training set might help


3) Fit test set well on cost function
   - Use of a bigger development set might help


4) Performs well in real world
   - Development test set is not set correctly
   - Cost function is not evaluating the right thing

## Single Number Evaluation Metric

Single number evaluation metric can speed up choosing a classifier  
F1-score combines both precision and recall (a harmonic mean)
- Optimizing metric : metric has to be optimized as much as possible
- Satisficing metric : metric has to meet expectation set

## Train/Dev/Test Set

- Choose dev/test set to reflect future data and important data
- Set up test set size that can give a confidence
- Test set could be less 30% of the whole data set
- Dev set has to be big enough to evaluate different ideas
- Evaluation metric that helps better rank order classifiers
- Optimize the evaluation metric

## Human Level Performance

Human-level error gives an estimate of Bayes error

Bayes optimal error is defined as the best possible error  
Human-level performance is close to Bayes optimal error for natural perception problem

Tools that can improve ML before surpassing human level performance
- Get labeled data from humans
- Gain insight from manual error analysis: Why did a person get this right?
- Better analysis of bias/variance.

Problems where machine learning significantly surpasses human-level performance  
(especially with structured data)
- Online advertising
- Product recommendations
- Logistics (predicting transit time)
- Loan approvals

The two fundamental assumptions of supervised learning
1) You can fit the training set pretty well
2) The training set performance generalizes pretty well to the dev/test set

Human-level  
→ Avoidable bias
- Train bigger model
- Train longer, better optimization algorithms
- Neural Networks architecture/hyperparameters search

Training error  
→ Variance
- More data
- Regularization
- Neural Networks architecture/hyperparameters search

Development error

## Error Analysis

Error analysis process
1) Find a set of mislabeled examples in dev or test set
2) Look at the mislabeled examples for false positives and false negatives
3) Count up the number of errors that fall into various different categories
4) Prioritize improvement methods

## Cleaning Up Incorrectly Labeled Data

Correcting incorrect train set examples
- If the errors are reasonably random, it's probably okay to leave the errors

Correcting incorrect dev/test set examples
- Apply same process to dev and test sets (keep same distribution)
- Train and dev/test set may come from slightly different distributions
- Consider examining examples that is right but actually wrong (not easy)

## Build System Quickly, Then Iterate

Do not overthink or not make your first system too complicated  
Just build something quick and dirty and prioritize improvement methods

1) Set up development/ test set and metrics
   - Set up a target


2) Build an initial system quickly
   - Train training set quickly: Fit the parameters
   - Development set: Tune the parameters
   - Test set: Assess the performance


3) Use Bias/Variance analysis & Error analysis to prioritize next steps

## Mismatched Training and Dev/Test Set

Choose dev set and test set to reflect data expected to get

Training-Dev set is the data from training set but not trained  
Training-Dev set distinguishes variance/data mismatch problems

Bayes error (Human level error)  
	→ Avoidable Bias  
Training set error  
	→ Variance  
Training - Development set error  
	→ Data mismatch  
Development set error  
	→ Degree of overfitting to the development set  
Test set error

To address data mismatch
- Perform manual error analysis
- Understand the error differences between training and dev/test sets
- Make training data or collect data similar to dev/test sets  
    (Artificial data synthesis can help)

## Transfer Learning

Transfer learning is using neural network knowledge for another application

When to use transfer learning
- Task A and B have the same input 𝑥
- A lot more data for Task A than Task B
- Low level features from Task A could be helpful for Task B

Pre-training : Pre-initialize the weights of neural network with data A  
Fine-tuning : Updates all the weights afterwards with data B

Guideline
- Delete last layer of neural network
- Delete weights feeding into the last output layer of the neural network
- Create a new set of randomly initialized weights for the last layer only
- New data set (𝑥, 𝑦)

## Multi-task Learning

Multi-task learning is having one neural network do simultaneously several tasks

When to use multi-task learning
- Training on a set of tasks that could benefit from having shared lower-level features
- Usually: Amount of data you have for each task is quite similar
- Can train a big enough neural network to do well on all tasks

Cost can be computed just fine even when some entries not labeled

Multi-task learning is used much less often than transfer learning  
Computer vision object detection is one exception

## End-to-end Deep Learning

E2E DL is the simplification of a processing or learning systems into one NN

Requires enough data to learn a function of the complexity needed to map x and y
- The traditional way - small data set
- The hybrid way - medium data set
- The End-to-End deep learning way – large data set

Pros
- Go beyond human preconceptions by letting the data speak
- Less hand-designing of components needed

Cons
- Large amount of labeled data is required
- Excludes potentially useful hand-designed component

Areas using E2E deep learning
- audio transcripts
- image captures
- image synthesis
- machine translation
- steering in self-driving cars