function forward_propagation_fcl(layer::FullyConnectedLayer, input) 
    layer.input = input
    x = dot_product(input, layer.weights)
    layer.output =  x + layer.bias

    return layer.output
end

function backward_propagation_fcl(layer::FullyConnectedLayer, output_error, learning_rate::Float64) 
    input_error = dot_product(output_error, transpose(layer.weights))
    weights_error = dot_product(transpose(layer.input),  output_error)

    layer.weights -= learning_rate * floor.(weights_error, digits=8)
    layer.bias .-= learning_rate * floor.(output_error, digits=8)

    return input_error
end

function dot_product(A, B)
    col1 = size(A, 1)
    col2 = size(B, 1)
    if col1 < col2 || col1 == 1 || col2 == 1
        return A * B
    else
        val = sum(A.*B, dims=2)
        len = length(val)
        val2 = resize!(vec(val), len)
        if len > 1
            return permutedims(vcat(val2...)) 
        end      
        return val2
    end
end