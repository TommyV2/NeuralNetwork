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

function mean_squared_error′(y_true, y_pred)
    val = 2*(y_pred-y_true)/length(y_true)
    return val
end

a = [2 3]
b = [4 4]
c = [4]
d = [2]

A = -0.02315588799837393
B = [-0.02 0.01 0.03]

C = [-0.01199942; 0.00899976; 0.00299999]
D = -0.02315589

myC = [-0.011999424033175667; 0.008999757007872942; 0.0029999910000324;;]

outAB = [ 0.00046312 -0.00023156 -0.00069468]
outCD = [2.77857319e-04; -2.08397365e-04; -6.94674556e-05]

myAB = [-0.0004631177599674786]
myCD = [0.0002778573189572121 -0.00020839736528688673 -6.946745559288005e-5]

print(dot2(C,D))

