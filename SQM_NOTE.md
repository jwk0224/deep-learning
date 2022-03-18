# 1. Recurrent Neural Networks

## Examples of Sequence Data

Speech recognition  
Music generation  
Sentiment classification  
DNA sequence analysis  
Machine translation  
Video activity recognition  
Named entity recognition

## Representing Words

A sentence can be represented as sequence of words each of which is one-hot vector

Dictionary(= vocabulary) : a list of the words used for representations
- commercial applications : size of 30,000~100,000 words
- large internet companies : size of more than a million words

One-hot vector
- a vector with 1 in position of representing word from dictionary and 0 everywhere else

## Recurrent Neural Network Model

RNN can deal with problems that standard neural network cannot
- Inputs, outputs can be different lengths in different examples
- Doesn't share features learned across different positions of text

RNN scans through the data from left to right
- The parameters it uses for each time step are shared
- Later steps use information from previous steps as an input

t : time step (= each word)  
a<0> : vector of zeros  
x<0> : vector of zeros

Activation for each time step:
- a< t > = g( Wa * [ a< t-1 >, x< t > ] + ba )

Prediction for each time step:  
- yhat< t > = g( Wy * a< t > + by )

Loss for each time step in forward/back propagation:  
- L< t >(yhat< t >, y< t >) = -y< t > * log yhat< t > - (1 - y< t >) * log(1 - yhat< t >)

Loss for entire time step in forward/back propagation:  
- L(yhat, y) = ∑_t L< t >(yhat< t >, y< t >)

## Different Architectures of RNNs

One-to-One
- standard neural network

One-to-Many
- ex. Music generation

Many-to-One
- ex. Sentiment classification

Many-to-Many (Tx = Ty)
- ex. Named entity recognition

Many-to-Many (Tx ≠ Ty)
- encoder + decoder
- ex. Machine translation

## Language Model

Language model estimates the probability of a particular sentence
- ex. speech recognition, machine translation and etc
- RNNs are very good at language modeling

To build a language model,

1) Prepare a training set
   - a large corpus(= large set of sentences)


2) Tokenize sentence
   - map each word to dictionary making one-hot vector
   - EOS token : optional extra token appended to end of sentence
   - UNK token : optional extra token for unknown words
   - punctuation : it's optional to include punctuation as a token


3) Build an RNN model
   - define the cost function and train RNN on a large training set
   - RNN learns to predict one word at a time going from left to right


4) Predict probability of sentence
   - get probability of sentence by multiplying probability of each word

Sampling a sequence from a trained RNN
1) use the probabilities output of softmax layer of RNN to randomly sample a word for that time step
2) pass the selected word to the next time step
3) repeat above process until it reaches the last time step
   - when EOS token is sampled
   - when reaches manually picked number of time step
- ignore UNK token and re-sample when having unknown word is not desired

Character level language model (vs Word level)
- no worry for unknown word tokens
- not good for capturing long range dependencies due to longer sequences
- computationally expensive to train
- used for specialized applications (word level is more common)

## Vanishing Gradients Problem with RNNs

When training a very deep neural network,  
as a function of the number of layers,
- derivative can grow exponentially : exploding gradient problem (showing NaN = overflow)
- derivative can decrease exponentially : vanishing gradient problem

Exploding gradients can be solved by
- gradient clipping : re-scale gradient vectors if it is bigger than threshold

Vanishing gradients can be solved by
- Gated Recurrent Unit (GRU)
- Long Short Term Memory (LSTM)

By solving vanishing gradients problem,  
the model can capture much longer range dependencies

## Gated Recurrent Unit (GRU)

Gated recurrent unit is a gating mechanism in RNN  
to learn long range connections in a sequence

The cell remembers values over time steps,  
and at every time step, the update gate is going to consider overwriting cell information

GRU unit is composed of
- cell state (c) : memory that gets passed onto future time steps
- hidden state (a) : value used to predict y of current time step and passed onto next time steps
- update gate (𝚪u) : decides whether to update c< t > or not
- relevance gate (𝚪r) : tells the relevance of c< t-1 > in computing c< t >

## Long Short Term Memory (LSTM)

Long short term memory unit is a gating mechanism in RNN  
to learn long range connections in a sequence

The cell state remembers values over time steps,  
and at every time step, the three gates regulate the flow of information into and out of the cell

LSTM unit is composed of

- cell state (c) : memory that gets passed onto future time steps
	- c< t > = 𝚪f< t > * c< t-1 > + 𝚪u< t > * cc< t >
	- a value is decided by forget gate and update gate


- candidate value (cc) : information from current time step that may be stored in c< t >
	- cc< t > = tanh(Wc [a< t-1 >, x< t >] + bc)
	- a tensor containing values that range from -1 to 1


- hidden state (a) : value used to predict y of current time step and passed onto next time steps
	- a< t > = 𝚪o< t > * tanh(c< t >)
	- a value is decided by output gate


- forget gate (𝚪f) : decides whether to remember value in c< t-1 >
    - 𝚪f< t > = sigmoid(Wf [a< t-1 >, x< t >] + bf)
    - a tensor containing values between 0 and 1
      - close to 0 : forget stored value in c< t-1 >
      - close to 1 : remember stored value in c< t-1 >


- update gate (𝚪i) : decides what parts of value in cc< t > are passed onto c< t >
    - 𝚪i< t > = sigmoid(Wi [a< t-1 >, x< t >] + bi)
    - a tensor containing values between 0 and 1
      - close to 0 : prevent value in cc< t > from being passed onto c< t >
      - close to 1 : allows value in cc< t > to be passed onto c< t >


- output gate (𝚪o) : decides output activation of current time step
	- 𝚪o< t > = sigmoid(Wo [a< t-1 >, x< t >] + bo)
	- a tensor containing values between 0 and 1

GRU is relatively recent invention and a simplification of more complicated LSTM model
- LSTM is more powerful and more flexible since there's three gates instead of two
- GRU is simpler and easier to build a much bigger network since there's only two gates
- There isn't an universally superior algorithm
- LSTM is proven default choice, GRU has been gaining a lot of momentum

## Bidirectional RNN

Unidirectional RNN computes only forward direction

Bidirectional RNN computes both forward and backward direction  
So it takes into account information from the past and from the future

Bidirectional RNN with a LSTM is commonly used default choice for a lot of NLP problems

Disadvantage of the bidirectional RNN is that  
the entire sequence of data is needed before making predictions  
- ex. need to wait for the person to stop talking to make speech recognition prediction

## Deep RNN

Stacking multiple layers of RNNs together is sometimes useful to learn very complex functions

As deep RNNs are quite computationally expensive to train,  
deep RNN layers are often followed by deep conventional neural networks

# 2. Natural Language Processing & Word Embeddings

## Word Embeddings

Words are represented by a high dimensional feature vectors
- words are 'embedded' to a point in high dimensional space
- it's better than one-hot vector for representing different words
- it has been one of the most important ideas in NLP

Most common sizes for word vectors range between 50 and 1000  
We cannot guarantee that the individual components of the embeddings are interpretable

t-SNE is common algorithm for visualizing word embeddings
- it maps embeddings non-linearly to a lower dimensional space (ex. plotting 2D data)

Terminology
- embedding = encoding
- one-hot vector = sparse vector
- feature vector = dense vector

## Transfer Learning Using Word Embeddings

Featurized representations of different words by word embeddings  
can be plugged into many NLP applications

1. Learn word embeddings from large text corpus (1~100B words)
   - (or download pre-trained embedding online)


2. Transfer embedding to new task with smaller training set (say, 100k words)
   - use relatively lower dimensional feature vectors


3. Optional: Continue to finetune the word embeddings with new data
   - only if task 2 has a pretty big data set

Transfer learning is most useful (when transfer from task A to task B)  
when there is a large data set for A and a relatively smaller data set for B

NLP tasks that using transfer learning (using word embeddings) is

- useful (a smaller data set for B)
	- named entity recognition
	- text summarization
	- co-reference resolution
	- parsing

- not useful (a larger data set for B)
	- language modeling
	- machine translation

## Analogy Reasoning Using Word Embeddings

Man -> King  
Woman -> ? (= Queen)

Word embeddings learning algorithm running on the large text corpus  
can be used for analogy reasoning (spotting patterns between words)
- 30% to 75% accuracy for exact match
- analogy reasoning by itself may not be the most important NLP application
- but it can help convey a sense of what word embeddings are doing

1) Measure the similarity between two different word embeddings
2) Find a word that maximize the similarity

Word w = argmax( sim( e_w, e_king - e_man + e_woman ) )
- sim : cosine similarity is most commonly used
	- range from -1 to 1
	- -1 : exactly opposite
	- 0 : orthogonality or decorrelation
	- 1 : exactly the same

Because t-SNE is non-linear mapping,  
many of the parallelogram analogy relationships will be broken by t-SNE

## Embedding Matrix

Algorithm to learn word embeddings learns embedding matrix

- Embedding matrix (E) : feature count x vocabulary count
- One-hot vector (o) : vocabulary count  x 1
- Embedding vector (e) : feature count x 1

E * o_j = e_j = embedding for word j

## Neural Language Model for Word Embeddings Learning

Language model algorithm is one of the earlier and pretty successful way to learn word embeddings

Given previous words, the model predicts the next word in text corpus
- Context : previous few words
- Target : next word
- Parameters
	- matrix E
	- neural network weights
	- softmax weights

1) take previous few words from sentence and make one-hot vectors
2) get embedding vectors by multiplying them by embedding matrix E
3) stack them into a vector
4) pass it to a neural network
   - the input size can be fixed by using fixed historical window (ex. using fixed 4 previous words)
5) pass it to a softmax classifier that outputs Y-hat
   - probabilities of being a next word of entire vocabulary 

## Word2Vec for Word Embeddings Learning

Word2Vec algorithm is simpler and computationally more efficient way to learn word embeddings

Skip-gram model
- takes one word as input and predicts some words skipping a few words from the left or the right side

Given the context word, the model predicts a randomly picked word within some window
- Context : a randomly picked word
- Target : a randomly picked word within some window
- Parameters
	- matrix E
	- softmax weights

1) sample a word from sentence and make one-hot vector
   - sampling uniformly random ends up picking frequently occurring words (ex. the, of, a, and, to)
   - heuristics are used to balance out something from the common words together with the less common words
2) get an embedding vector by multiplying it by embedding matrix E
3) pass it to a softmax classifier that outputs Y-hat
   - probabilities of being a randomly picked target word of entire vocabulary
   - a hierarchical softmax classifier is used to increase computational speed

CBOW(Continuous Bag of Words) model
- takes surrounding contexts from middle word and uses the surrounding words to predict the middle word

## Negative Sampling for Word Embeddings Learning

Negative sampling algorithm is much more efficient way than skip-gram model to learn word embeddings

Given a pair of words, the model predicts whether a pair is positive or negative example
- Context : a randomly picked word
- Target : a randomly picked word
- Label : positive (target word is from text within some window), negative (target word is from dictionary)
- Parameters
	- matrix E
	- logistic regression weights

1) sample a word from sentence and make one-hot vector
2) get an embedding vector by multiplying it by embedding matrix E
3) pass it to binary logistic regression classifiers 
4) repeat above k + 1 (out of dictionary size) times per iteration
	1) sample a context word for input
	2) sample a word within some window for classifier
	3) label 1 for a positive example
	4) take a same sampled context word for input
	5) sample a word within a dictionary for classifier
		- sample according to frequency (very high representation of the, of, and, ...)
		- sample uniformly at random (non-representative of the distribution of words)
		- sample power of 3/4 of frequency heuristically (best)
	6) label 0 for a negative example
	7) repeat negative example k times
		- k = 5~20 for smaller data set
		- k = 2~5 for larger data set

## GloVe for Word Embeddings Learning

GloVe algorithm is even simpler way than other algorithms to learn word embeddings
- GloVe = Global Vectors for word representation
- not used as much as the Word2Vec or the skip-gram but has some momentum due to simplicity

minimize  ∑_i ∑_j f(X_ij) ( θ_i^T * e_j + b_i + b'_j - logX_ij   )^2
- Just minimizing a square cost function learns meaningful word embeddings

θ_i, e_j : should be initialized randomly at the beginning of training

X_ij : number of times j (=target) appears in context of i (=context)
- how related are words i and j as measured by how often they occur with each other

f(X_ij) : heuristic weighting factor 
- sum only over the pairs of words that have co-occurred at least once (0 if X_ij = 0)
- gives a meaningful amount of computation even to the less frequent words
- gives more weight but not an unduly large amount of weight to words like this, is, of, a, ...

## Sentiment Classification

Good sentiment classifiers can be built with word embeddings  
even with only modest-size label training sets

If word embeddings were trained from a large data set,  
words not in label training set can be generalized well

Simple sentiment classification model

1) take words from sentence and make one-hot vectors
2) get embedding vectors by multiplying them by embedding matrix E
3) sum or average embedding vectors
4) pass it to a softmax classifier that outputs Y-hat (ex. 5 stars)

- by using the average operation, it works for different length of words (short or long review)
- ignoring word order is the problem
	- ex. Completely lacking in good taste, good service, and good ambiance (lacking < good x 3)

RNN sentiment classification model

1) take words from sentence and make one-hot vectors
2) get embedding vectors by multiplying them by embedding matrix E
3) feed them into an RNN to compute the representation at the last time step that predicts Y-hat
	- a many-to-one RNN architecture

- it takes word sequence into account much better

## Dibiasing Word Embeddings

Word embeddings can reflect gender, ethnicity, age, sexual orientation and other biases of the text used to train the model

Reducing or eliminating bias of learning algorithms is a very important problem  
because algorithms are being asked to help with or to make more important decisions in society

Addressing bias in word embeddings

1) Identify bias direction
	- use singular value decomposition algorithm (similar to PCA)
2) Neutralize : for every word that is not definitional, project to get rid of bias
	- a linear classifier can tell what words to pass through the neutralization step
	- most words in the English language are not definitional (ex. man/woman in gender)
3) Equalize pairs : adjust some pairs of words to have same distance from equalized words
	- the number of pairs to equalize can be relatively small (feasible to hand-pick)

# 3. Sequence Models & Attention Mechanism

# 4. Transformer Network