using Random

function get_weights(input_size::Int, output_size::Int, idx) #TO DELETE
    weights = rand(Float64, (input_size, output_size)) .- 0.5 
    #TO DELETE
    if idx == 1
        weights = [-0.02 0.01 0.03; 0.01 0.02 0.03]
    else
        weights = [-0.02; 0.01; 0.03;;]
    end
    return weights
end

function get_bias(output_size::Int, idx) #TO DELETE
    bias = rand(Float64, (1, output_size)) .- 0.5
    #TO DELETE
    if idx == 1
        bias = [-0.012 0.009 0.003]
    else
        bias = [-0.012;;]
    end

    return bias
end
