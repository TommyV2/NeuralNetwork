import numpy as np

def get_weights(input, output):
    weights =  np.random.rand(input, output) - 0.5

    return weights

def predict(input_data):
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            print(output)
            #output = forward_propagation(output)
            
def mse_prime(y_true, y_pred):
    print(y_true.size)
    return 2*(y_pred-y_true)/y_true.size

a = np.array([[2], [3], [5]])
b = np.array([4, 4])
c = np.array([4])
d = np.array([2])
print(a.T)

def mse(y_true, y_pred):  
    val = np.mean(np.power(y_true-y_pred, 2))
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

print(mse(np.array([0.12345]), np.array([0.1234])))
print(mse_prime(np.array([0.12345]), np.array([0.1234])))

