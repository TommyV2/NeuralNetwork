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


# A = 0 0 
# B = [2.5067575330553957 0.1295373142612433 0.288392041029594; 2.615823750138521 -0.1394371453592334 -0.33878987491871604]
# out = 
a = 30
a -= 4 * 5
print(a)
