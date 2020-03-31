# BearX
### What is BearX?
BearX is my personal try to create self-sufficient Deep-Learning library which contains the best aspects (in my opinion)
of both Keras and Pytorch frameworks, but most importantly I am working on this project to get in-depth knownledge about
maths behind neural network and learn how properly structure larger projects in Python.
### Quickstart with BearX
```python
from bearx.layers import Linear
from bearx.models import Sequential

model = Sequential()
model.add(Linear(2, 2), activation='relu')

# Before training we have to compile model
model.compile(loss='mse',
              optimizer='sgd')
```
### What have been implemented so far? (31.3.2020)
#### Models
1) Sequential
#### Layers
1) Linear (Dense)
2) RNN
3) Embedding (testing)
4) Flatten (testing)
#### Activations
1) Relu
2) Tanh
3) Sigmoid (testing)
4) Softmax(testing)
#### Losses
1) MSE
2) Cross-Entropy (testing)
#### Optimizers
1) SGD
#### Initializers
1) Zeros
2) Ones
3) Normal Distribution
4) Uniform Distribution
5) RNNinit  
### Support
BearX supports only python 3.x


### Planning Releases What will be added soon?
#### Layers
1) Convolutional
#### Optimizers
1) RMSprop
2) Adam
#### Regulaizers
