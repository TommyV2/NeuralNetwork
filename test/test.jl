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

a = [1 1]
x = [2 2]
b = [1 2 3; 2 4 6]
c = [1 2 3]
d = [1; 2; 3]
# dot:
# col1 < col2

# dot2: 
# col1 >= col2

println(dot(a,b))
println(dot(c,d))
println(dot(a,x))
println(dot(b,c))

new = dot(b,c)
old = a

println(dot(new,old))



println(size([4 1;1 0]))