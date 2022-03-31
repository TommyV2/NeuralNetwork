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
            



a = np.array([1, 1])
x = np.array([2, 2])
b = np.array([[1, 2, 3], [2, 4, 6]])
c = np.array([1, 2, 3])
d = np.array([[1],[2],[3]])
print(np.dot(a,b))
print(np.dot(c,d))
print(np.dot(a,x))
print(np.dot(b,c))
