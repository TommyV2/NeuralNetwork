using Random

function get_weights(input_size::Int, output_size::Int)
    weights = rand(Float64, (input_size, output_size)) .- 0.5 
    return weights
end

function get_bias(output_size::Int)
    bias = rand(Float64, (1, output_size)) .- 0.5
    return bias
end
