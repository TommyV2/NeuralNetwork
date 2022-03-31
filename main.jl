include("structures.jl")
include("methods/network_methods.jl")
include("methods/helper_functions.jl")

function placeholder() end

x_train = [0 0; 0 1; 1 0; 1 1]
y_train = [0; 1; 1; 0]

network = Network(Any[], placeholder, placeholder)

add_layer(network, FullyConnectedLayer(2, 3, 1)) #TO DELETE
add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))
add_layer(network, FullyConnectedLayer(3, 1, 2)) #TO DELETE
add_layer(network, ActivationLayer(compute_tanh, compute_tanh′))

use_loss_function(network, mean_squared_error, mean_squared_error′)
epochs = 1000
learning_rate = 0.1
fit(network, x_train, y_train, epochs, learning_rate)

out = predict(network, x_train)
print(out)
