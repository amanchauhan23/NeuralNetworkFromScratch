def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 10000, learning_rate = 0.01, verbose = True):
    loss_vs_e = []
    for e in range(epochs):
        error = 0
        
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)
            error += loss(y, output)
            # backward 
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
                
        error /= len(x_train)
        if e%19==0:
            loss_vs_e.append([e, error])
        if verbose:
            print(f"Epoch: {e+1}/{epochs}, error={error}")
    return loss_vs_e