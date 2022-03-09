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

Dictionary : a list of the words used for representations
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

# 3. Sequence Models & Attention Mechanism

# 4. Transformer Network