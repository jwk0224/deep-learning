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
3) Equalize pairs : adjust some pairs of words to have same distance from neutralized words
	- the number of pairs to equalize can be relatively small (feasible to hand-pick)

# 3. Sequence Models & Attention Mechanism

## Sequence to Sequence Models

Sequence to sequence modeling : to generate most likely sentence
- machine translation (sentence to sentence)
- image captioning (image to sentence)
- speech recognition (audio to sentence)

Language modeling : to generate random sentence
- synthesizing novel text

Sequence to sequence modeling  
= encoder network + decoder network(= language model)  
= conditional language modeling

To generate most likely sentence in sequence to sequence modeling,  
we need to use approximate search algorithm rather than greedy search algorithm
- it's not always optimal to just pick one word at a time
- total number of combinations of words in the English sentence is exponentially large

## Beam Search

Beam search is the most widely used algorithm to generate most likely sentence

Beam search runs much faster but is not guaranteed to find exact maximum possibility  
compared to exact search algorithms like BFS (Breadth First Search), DFS (Depth First Search)

Beam search considers multiple alternatives at the time
- arg max Π_t=1~Ty P( y< t > | x, y<1>, y<2>, ... , y< t-1 > ) 
- parameter B (beam width) : number of alternatives to consider
	- large B : better result, slower
	- small B : worse result, faster
	- production system : B = 10~100
	- research system : B = 1,000~3,000
	- when B is very large, performance gain tends to diminish

1) take an input from encoder network
2) pass it to a softmax classifier
3) compute possibilities of every word being a next word 
4) take top B words and store them on memory
5) pass previous sequence to a softmax classifier
6) compute possibilities of every word being a next word 
7) take top B sequence of words and store them on memory
8) repeat 5~7 until getting an EOS output

Length normalization

- arg max ∑_t=1~Ty log P( y< t > | x, y<1>, y<2>, ... , y< t-1 > ) 

  - multiplying a lot of numbers less than one results in a very small number
  - it may cause numerical under-floor
      - too small for floating point of representation in computer to store accurately
  - by taking logs, maximize sum of log of probabilities instead of maximizing product of probabilities
  

- 1/Ty^α * ∑_t=1~Ty log P( y< t > | x, y<1>, y<2>, ... , y< t-1 > ) 

  - adding log of probability which is always less than or equal to 1 results in more negative number
  - objective function unnaturally tends to prefer very short outputs
      - because the probability of a short sentence is higher
  - by normalizing by the number of words, takes average of log of probability of each word
      - parameter α : 
          - 0 : no normalization
          - 1 : full normalization
          - 0.7 : heuristic softer approach in practice

## Error Analysis in Beam Search

Carry out error analysis to figure out  
what fraction of errors is due to beam search versus the RNN model

Let RNN compute P(y* | x) and P(yhat | x)
- y* : good output written by a human
- yhat : beam search output
- when using length normalization : evaluate optimization objective instead of probability above

Case 1 : P(y* | x) > P(yhat | x)
  - beam search chose yhat
  - but y* attains higher P(y | x)
  - conclusion : beam search is at fault
  - possible solution : increase the beam width
  
Case 2 : P(y* | x) <= P(yhat | x)
  - y∗ is a better output than yhat
  - but RNN predicted P(y* | x) < P(yhat | x)
  - conclusion : RNN model is at fault
  - possible solution :
      - add regularization
      - get more training data
      - try a different network architecture
      - others

## BLEU Score

BLEU score is useful single real number evaluation metric for text generation task
- BLEU : BiLingual Evaluation Understudy
- conventional way to evaluate a machine translation system
- not used for speech recognition because there's usually one ground truth

Combined BLEU score = BP * exp (1/n * ∑_n p_n)

BP : Brevity Penalty
- adjustment factor that penalizes translation systems if output translation is too short
- 1 if MT_output_length > reference_output_length
- exp(1 − reference_output_length/MT_output_length) otherwise

p_n : BLEU score on n-gram
- measures the degree to which MT output is similar or overlaps with reference outputs
- p_n = 1 if MT output is exactly equal to one of reference output
- p_n =  ∑_(ngram ∈ yhat) count_clip(ngram) / ∑_(ngram ∈ yhat) count(ngram)
  1) identify ngrams in MT(Machine Translation) output (= yhat)
  2) compute count_clip(ngram)
      1) count appearance of each ngram in each reference output
      2) sum each ngram count that is maximum among reference outputs
  3) compute count(ngram)
      1) count appearance of each ngram in MT output
      2) sum each ngram count
  4) divide count_clip(ngram) by count(ngram)

## Attention Model

Attention algorithm learns where to pay attention, so performs well for long input
- encoder-decoder model performs worse for text longer than 30 or 40 words
- attention algorithm solves this and is one of the most influential ideas in deep learning

Encoder unit
- hidden state : a<t'>
- input : x< t >
- output : connected to s< t >

 Decoder unit
- hidden state : s< t >
- input : previous hidden state s< t-1 >, context c< t > (= ∑ α<t, t'> x a<t'>)
- output : y< t >

Attention model computes a set of attention weights
- α<t, t'> = exp(e^<t, t'>) / ∑_t' exp(e^<t, t'>)
  - amount of attention y< t > should pay to a<t'>
  - sum of weights over t' is 1 by using softmax

One hidden layer neural network is trained to compute attention weights
- s< t-1 >, a<t'> -> hidden layer -> e<t, t'>

Attention model takes quadratic time to run
- a total number of attention parameters is input unit count x output unit count
- input and output are usually not too long in machine translation application

Application example
- machine translation : pay attention to certain part of text to generate text
- image captioning : pay attention to certain part of image to write caption
- date normalization : ex. July 20th 1969 -> 1969-07-20

## Speech Recognition

Sequence-to-sequence models are applied to audio data, such as the speech

Spectrogram : a common pre-processing step generated from raw audio clip data
- horizontal axis : time
- vertical axis : frequencies
- intensity of different colors : amount of energy

End-to-end deep learning : builds systems that input an audio clip and directly output a transcript
- using hand-engineered representations like phonemes is no longer needed
- one of the things that made this possible was much larger data sets

1) Attention model


2) CTC cost for speech recognition
   - CTC : Connectionist Temporal Classification
   - usually input time steps is much bigger than output time steps
   - CTC cost function collapses repeated characters not separated by blank
   - a bunch of blank characters ends up with a much shorter output text transcript

## Trigger Word Detection

Trigger word detection is the technology  
that allows devices like Amazon Alexa to wake up upon hearing a certain word

There is no wide consensus yet on what's the best algorithm for trigger word detection

Example of a trigger word detection algorithm
1) take an audio clip
2) compute spectrogram features
3) generates audio features x and pass through an RNN
4) define the target labels y in training set
   - set target labels to be 0 for every unit before saying the trigger word
   - when just finished saying the trigger word, set target label to be 1
   - set target labels to be 1 for several times or a fixed period of time before reverting back to 0
       - to make the model easier to train (training set is imbalanced with a lot more 0s than 1s)

# 4. Transformer Network

## Transformer Network

- Transformer is a relatively complex neural network architecture
- Transformer is an architecture that has completely taken the NLP world by storm
- many of the most effective algorithms for NLP today are based on Transformers

RNN -> GRU -> LSTM
- to capture long range dependencies and sequences, models have become more complex
- as complexity of sequential model increases, each unit could become a bottle neck
- all previous units need to be computed one by one to compute the output of final unit

Transformer architecture
- computes an attention of each word in the sentence in parallel for multiple times
	- not compute an attention of each word in the sentence from left to right
- attention based representations + CNN style of processing
	- Self-attention : an attention based way of representing words in the sentence in parallel
	- Multi-head Attention : a for loop over the self-attention process

## Self-Attention

Transformers self-attention mechanism finds the most appropriate representation for the word
- given a word, its neighbouring words are used to compute its context
  - by summing up the word values to map the Attention related to that given word
- it is much more nuanced, much richer representation for the word than fixed word embedding

Attention-based vector representation of a word
- A(q, K, V) = ∑_i ( exp(e^< q * k^< i > >) / ∑_j exp(e^< q * k^<j> >) ) * v< i >
- Attention(Q, K, V) = softmax( Q * K^T / sqrt(d_k) ) * V

Query(Q) : interesting questions about the words in a sentence
- q< i > = W_Q x x< i >

Key(K) : qualities of words given a Q
- k< i > = W_K x x< i >

Value(V) : specific representations of words given a Q
- v< i > = W_V x x< i >

W_Q, W_K, W_V are learning parameters

## Multi-head Attention

Self-attention(= head)
- computes a vector representation of a word by asking a question for each word

Multi-head attention mechanism
- does self-attention computation multiple times in parallel (asks multiple questions)
	- each head is independently computed
- learns a much richer, much better representation for each word

MultiHead(Q, K, V) = concat(head1, head2, ... headh) * Wo
- concatenation of attention values is used to compute the output of multi-headed attention
- headi = Attention(Wi_Q * Q, Wi_K * K, Wi_V * V)
- h = number of heads

## Transformer Architecture

Main ideas

1) pass embeddings of a sentence to encoder block
	- use SOS(Start Of Sentence), EOS(End Of Sentence) tokens


2) repeat encoder block N times (typical N=6)
	1) Multi-head attention layer
		- takes embeddings of input sentence and generate Q, K, V
		- compute a matrix that represents the sentence
	2) Feed forward neural network layer
		- helps determine what interesting features there are in the sentence


3) repeat decoder block N times
	1) Multi-head attention layer
		- takes embeddings of current generated output sentence and generate Q, K, V
		- compute a matrix that represents the sentence
	2) Multi-head attention layer
		- takes output of encoder block and generate K, V
			- context based on the input sentence
		- takes output of previous layer and generate Q
			- question based on the current generated sentence
		- compute a matrix that represents the sentence
	3) Feed forward neural network layer
		- predicts the next word for the output sentence


4) pass output of decoder block to linear and softmax layer to predict the next word

Positional encoding of the input
- unlike sequential models, transformer feeds data all at once so there is no order data
- the position of a word in the sentence can be extremely important to translation
- a unique position encoding vector made by a combination of sine and cosine is added to input embedding vector
    - the output of the encoding block contains
        - contextual semantic embedding
        - positional encoding information
- position encoding is also passed through the network with residual connections
    - to pass along positional information through the entire architecture

Add & Norm layer
- plays a role very similar to the batch norm and helps speed up learning
- used after every layer in the architecture

Masked multi-head attention
- during training time, entire correct output sentence is given
- hide some part of correct output sentence and let the network predict them
	- no need to generate the words one at a time during training

Since the paper 'Attention Is All You Need' came out,  
there have been many other iterations of this Transformer Network model like BERT, DistilBERT
