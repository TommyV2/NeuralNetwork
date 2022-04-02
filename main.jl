include("structures.jl")
include("methods/network_methods.jl")
include("methods/helper_functions.jl")

function placeholder() end

# x_train = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]
# y_train = [0.0; 1.0; 1.0; 0.0]
# x_test = [0.0; 1.0; 1.0; 0.0]

x_train = [0 0 0; 0 0 1; 0 1 0; 1 0 0; 1 1 0; 0 1 1; 1 0 1; 1 1 1]
y_train = [0; 1; 1; 1; 1; 1; 1; 0]
x_test = [0 0 0; 0 0 1; 0 1 0; 1 1 1]

network = Network(Any[], placeholder, placeholder)

add_layer(network, FullyConnectedLayer(3, 4, 1)) #TO DELETE
add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))
add_layer(network, FullyConnectedLayer(4, 1, 2)) #TO DELETE
add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))

epochs = 1000
learning_rate = 0.1

#train
use_loss_function(network, mean_squared_error, mean_squared_error′)
train_network(network, x_train, y_train, epochs, learning_rate)

# for layer in network.layers
#     type = typeof(layer)
#     if type == FullyConnectedLayer
#         w = layer.weights
#         b = layer.bias
#         println("$w")
#         println("$b")
#         println("---------------------")
#     end
# end

#test
out = predict(network, x_test)
print(out)
