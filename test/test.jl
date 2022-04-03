using Random
using LinearAlgebra
using Statistics

function get_weights(input_size::Int, output_size::Int)
    weights = rand(Float64, (input_size, output_size)) .- 0.5

    return weights
end

function get_bias(output_size::Int)
    bias = rand(Float64, (1, output_size)) .- 0.5

    return bias
end

function compute_tanh(x)
    val = tanh.(x)
    return val
end

function compute_tanh′(x)
    val = 1 - tanh.(first(x))^2
    return val
end

a = 3
b = 2
# wei = get_weights(3,3)
# println(wei)
# bias = get_bias(3)
# println(bias)
# input_data = [1 2 3]
# println(input_data)

# output = input_data'*wei #+ bias
# print(tanh(1))

b = [2, 1]

x = [1 2 3; 4 5 6]

function dot2(A, B)
    val = sum(A.*B, dims=2)
    len = length(val)
    val2 = resize!(vec(val), len)
    if len > 1
        return permutedims(vcat(val2...))
    end
    return val2
end

function dot3(A, B)
    return A * B
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

function mean_squared_error(y_true, y_pred)
    val = first(mean((y_true.-y_pred).^2, dims=2))
    return val
end

function mean_squared_error′(y_true, y_pred)
    val = 2*(y_pred.-y_true)/length(y_true)
    return round(first(val), digits=7) 
end

# Python: 
# backward 
# 1
# OK
# 2
# output_error = array([[ 0.02790589,  0.10272589, -0.00012149]])
# layer.weigths = [array([0.11167431]), array([0.42586258]), array([-0.00048065])]
# input_error = array([[ 0.02856361,  0.10892545, -0.00012294]])
# layer.input = array([[-0.15174484,  0.23856991, -0.10852952]])
# weights_error = [array([-0.0388127]), array([0.06102047]), array([-0.02775925])]
# self.weight = [array([0.11555558]), array([0.41976053]), array([0.00229528])]
# self.bias = [array([0.02054919])]

# Julia
# backward 
# 1
# OK
# 2
# output_error = [0.027905894552422105 0.10641727941997066 -0.00012010791216548986]
# layer.weigths = [0.11167431; 0.42586258; -0.00048065;;]
# input_error = [0.028563614534668005 0.10892545098204964 -0.00012293876117155483]
# layer.input = [-0.15174484150993992 0.23856991368861294 -0.1085295223420643]
# weights_error = [-0.0388126970340288; 0.06102047153164014; -0.027759253151511426;;]
# self.weight = [0.11555558; 0.419760533; 0.002295276;;]
# self.bias = [0.020549195000000003;;]


# Python 
# output_error = [0.028563614534668005, 0.10892545098204964, -0.00012293876117155483]
# self.input = array([-0.15292592,  0.24325717, -0.10895867])
# output_activation_prime = [[ 0.02790589  0.10272589 -0.00012149]]

# Julia
# output_error = [0.028563614534668005 0.10892545098204964 -0.00012293876117155483]
# self.input = [-0.15292592 0.24325717 -0.10895867]

# output_activation_prime = [0.027905894552422105 0.10641727941997066 -0.00012010791216548986]

function compute_tanh′(x)
    val = tanh.(x).^2 
    val = val .* (-1)
    val .+= 1
    return val
end

output = compute_tanh′([-0.15292592 0.24325717 -0.10895867]) .* [0.028563614534668005 0.10892545098204964 -0.00012293876117155483]

print(output)
# tanh **2 [0.0230265  0.0569156  0.01177866]
# [0.9769735  0.9430844  0.98822134]