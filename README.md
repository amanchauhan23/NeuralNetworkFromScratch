# Neural Network Library
A simple yet powerful neural network library built from scratch; featuring backpropagation, gradient descent, activation functions, cost functions.

### Introduction
Using this library implemented a NN for multiclass classification on MNIST dataset.

### Try it
How to Run  
Clone the Repository:

```bash
# Copy code
git clone <repository-url>
cd <repository-directory>
```

 Install Required Packages: 
 - Numpy for Llibrary implementation
 - Keras for MNIST dataset
 - Plotly, Sklearn for performance metrics/Viz

Try:
```bash
python minst.py
```

**mnist.py**
```python
import numpy as np
from dense import Dense
from performance_metrics import calculate_metrics
from loss import mse, mse_prime
from keras.datasets import mnist
from keras.utils import np_utils
from helper import predict, train
from activations import Tanh, Sigmoid

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessor
def preprocess(x, y):
    # model expects col vector as input
    x = x.reshape(x.shape[0], 28*28, 1)
    x = x.astype("float32")/255 # grayscale intensity ranges from [0, 255]
    # one-hot encoding
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x, y

x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

# NN Network
network = [
    Dense(28*28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()
]

# train
loss_vs_e = train(network, mse, mse_prime, x_train, y_train, epochs=100, learning_rate=0.1)

# test
y_pred = np.zeros(y_test.shape[0], dtype=int)
y_true = np.zeros(y_test.shape[0], dtype=int)

for i, (x, y) in enumerate(zip(x_test, y_test)):
    output = predict(network, x)
    y_pred[i] = np.argmax(output)
    y_true[i] = np.argmax(y)

# Performance metrics
_, acc, F1 = calculate_metrics(y_true, y_pred, loss_vs_e)
print(f"Accuracy: {acc*100}%\nF1 Score: {F1}")
```
### Performance Metrics
After training, the model achieves the following performance metrics:   

**Accuracy: 91.03%**    
**F1 Score: 0.911**

### Loss over epochs
![alt text](<newplot (4).png>)

### Confusion matrix
![Confusion matrix](<newplot (5).png>)