include("methods/constructor_methods.jl")

mutable struct FullyConnectedLayer
    input::Any
    output::Any
    weights::Matrix{Float64}
    bias::Matrix{Float64}

    function FullyConnectedLayer(input::Int , output::Int)
        weights = get_weights(input, output)
        bias = get_bias(output)
        new(0, 0, weights, bias)
    end    
end

mutable struct ActivationLayer 
    input::Any
    output::Any 
    activation::Function 
    activation′::Function

    function ActivationLayer(activation::Function , activation_prime::Function)
        new(0, 0, activation, activation_prime)
    end   
end

mutable struct Network
    layers::Vector{Any}  
    loss::Function
    loss′::Function

    function Network(layers::Vector{Any}, loss::Function, loss′::Function)
        new(layers, loss, loss′)
    end   
end

function convert_to_matrix(arr)
    len = length(arr)
    if len > 1
        return permutedims(vcat(arr...))
    end
    return arr
end