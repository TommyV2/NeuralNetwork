using MLDatasets
using Images
using TestImages
# using Plots
include("structures.jl")
include("methods/network_methods.jl")
include("methods/helper_functions.jl")


function placeholder() end

# MNIST EXAMPLE: ---------------------------
train_x_raw, train_y_raw = MNIST.traindata()
test_x_raw, test_y_raw  = MNIST.testdata()

# to save images:
# for i = 1:20
#     save("images/test"*string(i)*".png", MNIST.convert2image(MNIST.testtensor(i)))
# end

train_x = []
train_y = []
for i = 1:1000
    push!(train_x, reshape(train_x_raw[:, :, i], 784))
    y = zeros(10)
    y[train_y_raw[i] + 1] = 1.0
    push!(train_y,y)
end
train_x = train_x'
train_y = train_y'

test_x = []
test_y = []
for i = 1:100
    push!(test_x, reshape(test_x_raw[:, :, i], 784))
    y = zeros(10)
    y[test_y_raw[i] + 1] = 1.0
    push!(test_y,y)
end
test_x = test_x'
test_y = test_y'


network = Network(Any[], placeholder, placeholder)

# Network 1:
# learning rate = 0.1
# epoch 10/100   error=17.892757483638245
# epoch 25/100   error=10.563332528435122
# epoch 50/100   error=6.456265929172283
# epoch 90/100   error=4.347389388116945
#
add_layer(network, FullyConnectedLayer(784, 100))
add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))
add_layer(network, FullyConnectedLayer(100, 10))
add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))

# Network 2:
# epoch 10/100   error=5.600691435288279
# epoch 25/100   error=2.2441759134388533
# epoch 50/100   error=1.118932339798652
# epoch 90/100   error=0.6041599370121681
#
# add_layer(network, FullyConnectedLayer(784, 100))
# add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))
# add_layer(network, FullyConnectedLayer(100, 50))
# add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))
# add_layer(network, FullyConnectedLayer(50, 10))
# add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))

epochs = 100
learning_rate = 0.1

#train
use_loss_function(network, mean_squared_error, mean_squared_error′)
train_network(network, train_x, train_y, epochs, learning_rate)

#test
out = predict(network, test_x)
for i = 1:10
    expected = argmax(test_y[i])[2] - 1
    result = argmax(out[i])[2] - 1
    score = out[i][result+1]
    println("-------------------------------------")
    println("Expected number: ", expected)
    println("Result number: ", result, " with score: ", score)
    println("All scores: ")
    println(out[i])
    println("-------------------------------------")
end
