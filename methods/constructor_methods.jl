using Random

function get_weights(input_size::Int, output_size::Int, idx) #TO DELETE # TO DO: fix
    weights = rand(Float64, (input_size, output_size)) .- 0.5 
    # TO DELETE
    if idx == 1
        weights = [0.30368407 -0.09039105 0.38600846 -0.09039105; 0.4835337  -0.37062517 -0.37932169 -0.09039105; 0.4835337  -0.37062517 -0.37932169 -0.09039105]
    else
        weights = [0.11167431; 0.42586258; -0.00048065; 0.38600846;;]
    end
    return weights
end

function get_bias(output_size::Int, idx) #TO DELETE # TO DO: fix
    bias = rand(Float64, (1, output_size)) .- 0.5
    #TO DELETE
    if idx == 1
        bias = [-0.15292592  0.24325717 -0.10895867 0.38600846]
    else
        bias = [0.0461268;;]
    end

    return bias
end

# [[ 0.30368407 -0.09039105  0.38600846]
#  [ 0.4835337  -0.37062517 -0.37932169]]
# [[-0.15292592  0.24325717 -0.10895867]]
# [[ 0.11167431]
#  [ 0.42586258]
#  [-0.00048065]]
# [[0.0461268]]
