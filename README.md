# BearX
## What is BearX?
BearX is my personal try to create self-sufficient Deep-Learning library which contains the best aspects (in my opinion)
of both Keras and Pytorch frameworks, but most importantly I am working on this project to get in-depth knownledge about
maths behind neural network and learn how to properly structure larger projects in Python.
## Quickstart with BearX
```python
from bearx.layers import Linear
from bearx.models import Sequential

model = Sequential()
model.add(Linear(2, 2), activation='relu')

# Before training we have to compile model
model.compile(loss='mse',
              optimizer='sgd')
```
## What have been implemented so far? (31.3.2020)
### Models
* Sequential
### Layers
* Linear (Dense)
* RNN
* (testing)
* (testing)
### Activations
* Relu
* Tanh
* Sigmoid (testing)
* Softmax(testing)
### **Losses**
* MSE
* Cross-Entropy (testing)
### Optimizers
* SGD
### Initializers
* Zeros
* Ones
* Normal Distribution
* Uniform Distribution
* RNNinit  
## Support
BearX supports only python 3.x  

  
## What will be added soon?
### Layers
- [ ] Convolutional
### Optimizers
- [ ] RMSprop
- [ ] Adam
### Regularizers
