# BearX
### Overview
I am currently working on this library to get in-depth knownledge about
maths behind neural networks and to learn how properly structure larger
projects in python. As 

### Quick start with BearX
```python
from bearx.layers import Linear
from bearx.models import Sequential

model = Sequential()
model.add(Linear(2, 2), activation='relu')

# Before training we have to compile model
model.compile(loss='mse',
              optimizer='sgd')
```

### Support
BearX supports only python 3.x
