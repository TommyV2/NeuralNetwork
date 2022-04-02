function forward_propagation_fcl(layer::FullyConnectedLayer, input) 
    layer.input = input
    x = dot(input, layer.weights)
    layer.output =  x + layer.bias
    o = layer.output
    return layer.output
end

function backward_propagation_fcl(layer::FullyConnectedLayer, output_error, learning_rate::Float64) 
    input_error = dot2(output_error, transpose(layer.weights))
    weights_error = dot2(transpose(layer.input),  output_error)

    layer.weights -= learning_rate * weights_error
    layer.bias .-= learning_rate * output_error

    return input_error
end

function dot(A, B)
    col1 = size(A, 1)
    col2 = size(B, 1)
    if col1 < col2
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

function dot2(A, B)
    col1 = size(A, 1)
    col2 = size(B, 1)
    if col1 < col2 || col1 == 1 || col2 == 1
        # println("A: $A, B: $B")
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