using Statistics

function compute_tanh(x)
    val = tanh.(x)
    return val
end

# function compute_tanh′(x)
#     val = 1 - tanh.(first(x))^2
#     return val
# end

function compute_tanh′(x)
    val = tanh.(x).^2 .* (-1) .+ 1
    return val
end

function mean_squared_error(y_true, y_pred)
    val = first(mean((y_true.-y_pred).^2, dims=2)) # TO DO: fix mean!
    return val
end

function mean_squared_error′(y_true, y_pred)
    val = 2*(y_pred.-y_true)/length(y_true)
    return val
end

# function mean_squared_error′(y_true, y_pred)
#     val = 2*(y_pred.-y_true)/length(y_true)
#     return round(first(val), digits=7) 
# end