include("activation_layer_methods.jl")
include("fully_connected_layer_methods.jl")

function add_layer(network::Network, layer)
    push!( network.layers, layer )
end

function use_loss_function(network, loss, loss′)
        network.loss = loss
        network.loss′ = loss′
end

function predict(network, input_data)
        result = []
        for value in input_data
            output = value
            for layer in network.layers
                type = typeof(layer)
                if type == FullyConnectedLayer
                    output = forward_propagation_fcl(layer, output)
                else
                    output = forward_propagation_al(layer, output)
                end
            end
            push!(result, output)
        end
        return result
end

function train_network(network, x_train, y_train, epochs, learning_rate)
    layers_num = length(network.layers)
    for i in 1:epochs
        err = 0
        local j = 1
        for row in eachrow(x_train)
            output = convert_to_matrix(row)
            for layer in network.layers
                type = typeof(layer)
                if type == FullyConnectedLayer
                    output = forward_propagation_fcl(layer, output)
                else
                    output = forward_propagation_al(layer, output)
                end 
            end
            
            err += network.loss(y_train[j], output)
            error = network.loss′(y_train[j], output)
            for layer in Iterators.reverse(network.layers)
                type = typeof(layer)
                if type == FullyConnectedLayer
                    error = backward_propagation_fcl(layer, error, learning_rate)
                else
                    error = backward_propagation_al(layer, error)
                end 
            end
            j += 1    
        end
        err /= layers_num
        println("epoch $i/1000   error=$err")
    end
end

function convert_to_matrix(arr)
    len = length(arr)
    if len > 1
        return permutedims(vcat(arr...))
    end
    return arr
end