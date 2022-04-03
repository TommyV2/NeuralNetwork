function forward_propagation_al(layer::ActivationLayer, input) 
    layer.input = input
    layer.output = layer.activation(layer.input)

    return layer.output
end

function backward_propagation_al(layer::ActivationLayer, output_error::Matrix{Float64}) 
    input_error = layer.activation′(layer.input) .* output_error

    return input_error
end